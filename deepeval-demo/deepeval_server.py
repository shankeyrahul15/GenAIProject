#!/usr/bin/env python3
"""
Deepeval FastAPI Sidecar Server
This runs separately from the Node.js server and provides LLM evaluation metrics.

Installation:
  pip install fastapi uvicorn deepeval

Usage:
  python deepeval_server.py
  # or
  uvicorn deepeval_server:app --reload --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Union
import logging
import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import DeepEval base class
from deepeval.models.base_model import DeepEvalBaseLLM

app = FastAPI(
    title="Deepeval Evaluation Service",
    description="FastAPI sidecar for LLM evaluation using Deepeval",
    version="1.0.0"
)

# Add CORS middleware to allow Node.js calls
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class EvalRequest(BaseModel):
    """Request body for evaluation.
    
    Properly separates user query, retrieved context, and model output for accurate metric scoring.
    """
    query: Optional[str] = None  # what the user asked
    context: Optional[List[str]] = None  # list of retrieved docs or source passages
    output: Optional[str] = None  # model's answer to be evaluated (REQUIRED for most metrics, not for conversation_completeness)
    provider: Optional[str] = None  # LLM provider: 'groq' or 'openai'
    metric: Optional[Union[str, List[str]]] = "faithfulness"  # metric(s) to evaluate - string, array, or "all"
    expected_output: Optional[str] = None  # required for contextual_* metrics
    messages: Optional[List[dict]] = None  # for conversation_completeness: list of {role, content}


class MetricResult(BaseModel):
    """Individual metric evaluation result"""
    metric_name: str
    score: Optional[float] = None
    explanation: Optional[str] = None
    error: Optional[str] = None


class EvalResponse(BaseModel):
    """Response with evaluation metrics"""
    results: List[MetricResult]  # Array of metric results
    # Legacy fields for backward compatibility (when single metric)
    metric_name: Optional[str] = None
    score: Optional[float] = None
    explanation: Optional[str] = None
    error: Optional[str] = None


class GroqModel(DeepEvalBaseLLM):
    """Custom Groq model wrapper for DeepEval compatibility."""
    
    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile"):
        """Initialize Groq client.
        """
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1"
        )
        self.model_name = model
        logger.info(f"Initialized Groq model: {model}")
    
    def load_model(self):
        """Load model - required by DeepEvalBaseLLM."""
        return self.client
    
    def generate(self, prompt: str, schema: Optional[object] = None) -> str:
        """Generate completion using Groq API.
        
        Args:
            prompt: The input prompt
            schema: Optional Pydantic model for structured output
        
        Returns:
            Generated text response or JSON string if schema provided
        """
        try:
            # Check if we need structured output
            if schema:
                # Request JSON format in the prompt
                json_prompt = f"{prompt}\n\nRespond with valid JSON only, no other text."
                
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that responds in JSON format."},
                        {"role": "user", "content": json_prompt}
                    ],
                    temperature=0.0,
                    response_format={"type": "json_object"}  # Enable JSON mode
                )
                
                content = response.choices[0].message.content
                
                # Parse and validate JSON against schema if it's a Pydantic model
                try:
                    import json
                    json_data = json.loads(content)
                    # If schema is a Pydantic model, validate and return instance
                    if hasattr(schema, 'model_validate'):
                        return schema.model_validate(json_data)
                    return content
                except Exception as json_err:
                    logger.warning(f"Failed to parse JSON response: {str(json_err)[:100]}")
                    return content
            else:
                # Regular text generation with neutral system message
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0
                )
                return response.choices[0].message.content
                
        except Exception as e:
            logger.error(f"Groq API error: {str(e)}")
            raise
    
    async def a_generate(self, prompt: str, schema: Optional[object] = None) -> str:
        """Async generate - for DeepEval compatibility."""
        return self.generate(prompt, schema)
    
    def get_model_name(self) -> str:
        """Return model name - required by DeepEvalBaseLLM."""
        return self.model_name
    
    def should_use_azure_openai(self) -> bool:
        """Check if using Azure - required by DeepEvalBaseLLM."""
        return False


class MetricEvaluator:
    """Enterprise-grade metric evaluation system with hybrid strictness approach.
    
    Uses strict_mode=False for natural LLM judgment, then applies custom post-processing rules:
    
    - Faithfulness: Natural LLM scoring + hallucination detection (caps score if output mentions 
      entities like 'Salesforce', 'CRM' not in context)
      
    - Answer Relevancy: Natural LLM scoring + definition enforcement (for "What is X?" questions,
      requires output to mention X and use definitional language like "is a/an")
      
    - Contextual Precision/Recall: Natural LLM scoring without additional rules
    
    This hybrid approach leverages model intelligence while catching common failure patterns.
    OpenAI models (like gpt-4o-mini) provide stricter base scoring than Groq models.
    """
    
    SUPPORTED_METRICS = {
        "faithfulness": "Evaluates if the output is faithful to the source context (hybrid: LLM judgment + hallucination detection)",
        "answer_relevancy": "Evaluates how relevant the answer is to the input question (hybrid: LLM judgment + definition enforcement)", 
        "contextual_precision": "Evaluates the precision of retrieval in RAG systems (natural LLM judgment, requires expected_output)",
        "contextual_recall": "Evaluates the recall of retrieval in RAG systems (natural LLM judgment, requires expected_output)",
        "conversation_completeness": "Evaluates if a conversation covers all necessary topics (requires conversational context)",
        "hallucination": "Detects hallucinations in LLM output compared to context (lower is better, 0 = no hallucinations)"
    }
    
    def __init__(self, api_key: str, model_name: str = "llama-3.3-70b-versatile", use_groq: bool = False):
        """Initialize the evaluator with API credentials.
        
        Args:
            api_key: API key for the LLM provider (OpenAI or Groq)
            model_name: Model to use for evaluation
            use_groq: Whether to use Groq API instead of OpenAI
        """
        if not api_key or api_key == "your-openai-api-key-here" or api_key == "your-groq-api-key-here":
            raise ValueError("Valid API key is required")
        
        self.model_name = model_name
        self.use_groq = use_groq
        
        if use_groq:
            # Use custom Groq model
            logger.info(f"Using Groq API with model: {model_name}")
            self.model = GroqModel(api_key=api_key, model=model_name)
        else:
            # Standard OpenAI
            os.environ["OPENAI_API_KEY"] = api_key
            logger.info(f"Using OpenAI API with model: {model_name}")
            from deepeval.models import GPTModel
            self.model = GPTModel(model=model_name)
    
    def validate_metric(self, metric_name: str) -> bool:
        """Validate if the requested metric is supported."""
        return metric_name.lower() in self.SUPPORTED_METRICS
    
    def create_test_case(
        self,
        query: Optional[str],
        context: Optional[List[str]],
        output: str,
        expected_output: Optional[str],
        messages: Optional[List[dict]] = None
    ):
        """Create a standardized test case for evaluation.
        
        Note: expected_output is kept as None unless explicitly provided to avoid
        automatic pass scenarios where metrics would compare output against itself.
        
        For conversation_completeness, pass messages list to create ConversationalTestCase.
        """
        # Handle conversational test case
        if messages is not None:
            from deepeval.test_case import ConversationalTestCase
            
            # Convert messages to list of dictionaries with role and content
            turns = []
            for msg in messages:
                turns.append({
                    "role": msg.get("role", "user"),
                    "content": msg.get("content", "")
                })
            
            return ConversationalTestCase(turns=turns)
        
        # Standard LLMTestCase for other metrics (UNCHANGED - this is the existing code)
        from deepeval.test_case import LLMTestCase
        
        # Ensure context is always a list for deepeval
        retrieval_ctx = context or []
        
        return LLMTestCase(
            input=query or "",  # user question
            actual_output=output,  # model response
            retrieval_context=retrieval_ctx,  # RAG/context
            expected_output=expected_output  # may be None for some metrics
        )
    
    def evaluate_faithfulness(self, test_case) -> tuple[float, str]:
        """
        Pure DeepEval faithfulness:
        - Uses truths/claims/verdicts from deepeval only.
        - No additional post-processing, thresholds, or penalties.
        """
        from deepeval.metrics.faithfulness.faithfulness import FaithfulnessMetric

        metric = FaithfulnessMetric(
            model=self.model,              # your DeepEvalBaseLLM or model name
            include_reason=True,           # let DeepEval generate the reason
            async_mode=False,              # keep sync in this server
            strict_mode=False,             # no hard clamp to 0 below threshold
            penalize_ambiguous_claims=False
        )

        score = metric.measure(test_case)      # DeepEval computes truths/claims/verdicts internally
        explanation = metric.reason or "Faithfulness (DeepEval core)."
        return score, explanation

    def evaluate_answer_relevancy(self, test_case) -> tuple[float, str]:
        """
        Pure DeepEval Answer Relevancy:
        - Uses DeepEval's native statements/verdicts/score.
        - No custom post-processing or caps.
        """
        from deepeval.metrics.answer_relevancy.answer_relevancy import AnswerRelevancyMetric

        metric = AnswerRelevancyMetric(
            model=self.model,        # DeepEvalBaseLLM or model name already init'd
            include_reason=True,     # let DeepEval generate the reason
            async_mode=False,        # keep server synchronous
            strict_mode=False        # no threshold clamp
        )

        score = metric.measure(test_case)
        explanation = metric.reason or "Answer Relevancy (DeepEval core)."
        return score, explanation

    def evaluate_contextual_precision(self, test_case) -> tuple[float, str]:
        """
        Pure DeepEval Contextual Precision:
        - Requires: input, expected_output, retrieval_context
        - No post-processing; returns DeepEval's score and reason.
        """
        from deepeval.metrics.contextual_precision.contextual_precision import ContextualPrecisionMetric

        metric = ContextualPrecisionMetric(
            model=self.model,
            include_reason=True,
            async_mode=False,
            strict_mode=False,
        )
        score = metric.measure(test_case)
        explanation = metric.reason or "Contextual Precision (DeepEval core)."
        return score, explanation

    def evaluate_contextual_recall(self, test_case) -> tuple[float, str]:
        """Pure DeepEval Contextual Recall (no post-processing)."""
        from deepeval.metrics.contextual_recall.contextual_recall import ContextualRecallMetric

        metric = ContextualRecallMetric(
            model=self.model,
            include_reason=True,
            async_mode=False,
            strict_mode=False,
        )
        score = metric.measure(test_case)
        return score, (metric.reason or "Contextual Recall (DeepEval core).")

    def evaluate_conversation_completeness(self, conv_case) -> tuple[float, str]:
        """Pure DeepEval Conversation Completeness (no post-processing).
        Expects a deepeval.test_case.ConversationalTestCase.
        """
        from deepeval.metrics.conversation_completeness.conversation_completeness import (
            ConversationCompletenessMetric,
        )

        metric = ConversationCompletenessMetric(
            model=self.model,
            include_reason=True,
            async_mode=False,
            strict_mode=False,
        )
        score = metric.measure(conv_case)
        return score, (metric.reason or "Conversation Completeness (DeepEval core).")

    def evaluate_hallucination(self, test_case) -> tuple[float, str]:
        """
        Pure DeepEval Hallucination (no post-processing).
        Requires: input, actual_output, context.
        Note: Lower is better. 0.0 = no hallucinations.
        """
        from deepeval.metrics.hallucination.hallucination import HallucinationMetric

        metric = HallucinationMetric(
            model=self.model,
            include_reason=True,
            async_mode=False,
            strict_mode=False,
        )

        # Compatibility: some pipelines set `retrieval_context` only.
        if getattr(test_case, "context", None) is None and getattr(test_case, "retrieval_context", None) is not None:
            try:
                setattr(test_case, "context", test_case.retrieval_context)
            except Exception:
                pass

        score = metric.measure(test_case)
        explanation = metric.reason or "Hallucination (DeepEval core; lower is better, 0 = none)."
        return score, explanation

    def evaluate(
        self,
        metric_name: str,
        *,
        query: Optional[str] = None,
        context: Optional[List[str]] = None,
        output: str = "",
        expected_output: Optional[str] = None,
        messages: Optional[List[dict]] = None
    ) -> tuple[float, str]:
        """Main evaluation method that routes to specific metric evaluators.
        
        Uses keyword-only arguments for better testability and clarity.
        Validates metric-specific requirements before calling DeepEval:
        - faithfulness: output required, context + query recommended
        - answer_relevancy: query + output required
        - contextual_precision/recall: context + output + expected_output required
        - conversation_completeness: messages required (list of conversation turns)
        - hallucination: output + context required, query recommended
        
        Args:
            metric_name: Which metric to evaluate
            query: User's question or input (optional for most metrics)
            context: List of retrieved documents or source passages (optional for some metrics)
            output: Model's generated response (required for most metrics)
            expected_output: Reference answer (required for contextual_* metrics)
            messages: List of conversation messages for conversation_completeness metric
            
        Returns:
            Tuple of (score, explanation)
            
        Raises:
            ValueError: If metric is unsupported or required fields are missing
        """
        metric_name = metric_name.lower()
        
        if not self.validate_metric(metric_name):
            raise ValueError(f"Unsupported metric: {metric_name}. Supported: {list(self.SUPPORTED_METRICS.keys())}")
        
        # Validate metric-specific requirements
        if metric_name in ["contextual_precision", "contextual_recall"]:
            # These metrics compare context vs expected answer
            if not expected_output:
                raise ValueError(f"{metric_name} requires 'expected_output' field")
            if not context:
                raise ValueError(f"{metric_name} requires 'context' field (list of retrieved passages)")
        
        if metric_name == "answer_relevancy":
            if not query:
                raise ValueError("answer_relevancy requires 'query' field (the user's question)")
        
        if metric_name == "conversation_completeness":
            if not messages:
                raise ValueError("conversation_completeness requires 'messages' field (list of conversation turns with role and content)")
        
        if metric_name == "hallucination":
            if not context:
                raise ValueError("hallucination requires 'context' field (list of source passages to check against)")
        
        # Create test case with proper structure
        test_case = self.create_test_case(
            query=query,
            context=context,
            output=output,
            expected_output=expected_output,
            messages=messages
        )
        
        # Route to appropriate evaluation method
        if metric_name == "faithfulness":
            return self.evaluate_faithfulness(test_case)
        elif metric_name == "answer_relevancy":
            return self.evaluate_answer_relevancy(test_case)  
        elif metric_name == "contextual_precision":
            return self.evaluate_contextual_precision(test_case)
        elif metric_name == "contextual_recall":
            return self.evaluate_contextual_recall(test_case)
        elif metric_name == "conversation_completeness":
            return self.evaluate_conversation_completeness(test_case)
        elif metric_name == "hallucination":
            return self.evaluate_hallucination(test_case)
        else:
            raise ValueError(f"Metric {metric_name} is not implemented yet")


def init_evaluator_from_env() -> MetricEvaluator:
    """Initialize MetricEvaluator from environment variables.
    
    Returns:
        Configured MetricEvaluator instance
        
    Raises:
        ValueError: If required API keys are missing
    """
    groq_api_key = os.getenv("GROQ_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    eval_model = os.getenv("EVAL_MODEL", "llama-3.3-70b-versatile")
    
    # Determine which API to use based on EVAL_MODEL
    # Groq models: llama-*, mixtral-*, gemma*, qwen*
    # OpenAI models: gpt-*
    use_groq = any(eval_model.startswith(prefix) for prefix in ["openai/","llama-", "mixtral-", "gemma", "qwen"])
    
    if use_groq:
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY environment variable is required when using Groq models")
        
        logger.info(f"Using Groq API for evaluation with model: {eval_model}")
        return MetricEvaluator(
            api_key=groq_api_key,
            model_name=eval_model,
            use_groq=True
        )
    else:
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required when using OpenAI models")
        
        # Override model if not a valid GPT model
        actual_model = eval_model if eval_model.startswith("gpt-") else "gpt-4o-mini"
        if actual_model != eval_model:
            logger.warning(f"EVAL_MODEL '{eval_model}' is not a valid GPT model, using '{actual_model}' instead")
        
        logger.info(f"Using OpenAI API for evaluation with model: {actual_model}")
        return MetricEvaluator(
            api_key=openai_api_key,
            model_name=actual_model,
            use_groq=False
        )


@app.post("/eval", response_model=EvalResponse)
async def evaluate_llm_response(req: EvalRequest):
    """
    Evaluate an LLM response using one or more metrics.
    
    Supports:
    - Single metric: metric="faithfulness"
    - Multiple metrics: metric=["faithfulness", "answer_relevancy"]
    - All metrics: metric="all"
    
    Each metric can be used independently to teach specific evaluation concepts.
    
    Args:
        req: EvalRequest with query, context, output, metric type(s), and optional provider
        
    Returns:
        EvalResponse with array of metric results
    """
    # Parse metric parameter - can be string, array, or "all"
    metric_param = req.metric or "faithfulness"
    
    # Convert to list of metrics
    if isinstance(metric_param, str):
        if metric_param.lower() == "all":
            # Get all supported metrics except conversation_completeness if no messages
            metrics_to_eval = list(MetricEvaluator.SUPPORTED_METRICS.keys())
            if not req.messages and "conversation_completeness" in metrics_to_eval:
                metrics_to_eval.remove("conversation_completeness")
        else:
            metrics_to_eval = [metric_param]
    else:
        metrics_to_eval = metric_param
    
    # Validate minimal fields for each metric
    for metric_name in metrics_to_eval:
        metric_name_lower = metric_name.lower()
        
        # For conversation_completeness, messages are required instead of output
        if metric_name_lower == "conversation_completeness":
            if not req.messages:
                raise HTTPException(
                    status_code=400, 
                    detail="messages field is required for conversation_completeness metric"
                )
        else:
            if not req.output:
                raise HTTPException(
                    status_code=400, 
                    detail=f"output field is required for {metric_name_lower} metric"
                )
    
    try:
        logger.info(f"=== Evaluation Request ===")
        logger.info(f"Metrics: {metrics_to_eval}")
        logger.info(f"Query: {req.query[:100] + '...' if req.query and len(req.query) > 100 else req.query or 'None'}")
        logger.info(f"Context items: {len(req.context) if req.context else 0}")
        logger.info(f"Output length: {len(req.output) if req.output else 0}")
        logger.info(f"Expected output: {'provided' if req.expected_output else 'None'}")
        logger.info(f"Messages: {len(req.messages) if req.messages else 0} turns")
        
        # Initialize evaluator from environment
        evaluator = init_evaluator_from_env()
        
        # Evaluate each metric
        results = []
        for metric_name in metrics_to_eval:
            try:
                score, explanation = evaluator.evaluate(
                    metric_name=metric_name,
                    query=req.query,
                    context=req.context,
                    output=req.output,
                    expected_output=req.expected_output,
                    messages=req.messages
                )
                
                results.append(MetricResult(
                    metric_name=metric_name,
                    score=score,
                    explanation=explanation
                ))
                
                logger.info(f"✓ {metric_name}: {score}")
                
            except ValueError as ve:
                # Metric-specific validation error
                logger.warning(f"✗ {metric_name}: {str(ve)}")
                results.append(MetricResult(
                    metric_name=metric_name,
                    score=None,
                    explanation=None,
                    error=str(ve)
                ))
            except Exception as e:
                # Unexpected error for this metric
                logger.error(f"✗ {metric_name}: {str(e)}")
                results.append(MetricResult(
                    metric_name=metric_name,
                    score=None,
                    explanation=None,
                    error=f"Evaluation failed: {str(e)}"
                ))
        
        # Build response with backward compatibility
        response = EvalResponse(results=results)
        
        # For backward compatibility: populate legacy fields with first successful result
        # This ensures existing clients that expect metric/score/explanation still work
        for result in results:
            if result.score is not None:  # First successful result
                response.metric_name = result.metric_name
                response.score = result.score
                response.explanation = result.explanation
                response.error = result.error
                break
        
        return response
    
    except Exception as e:
        # Unexpected errors (API failures, etc.)
        logger.exception("Evaluation error")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "service": "Deepeval Evaluation Service",
        "version": "1.0.0"
    }


@app.get("/example")
async def example_evaluation():
    """Smoke test endpoint that runs a fixed faithfulness check.
    
    Useful for CI/CD pipelines to verify the evaluation system is working correctly.
    Tests the full evaluation pipeline with a simple, predictable example.
    """
    try:
        # Fixed example: faithful output
        test_req = EvalRequest(
            query="What is Selenium?",
            context=["Selenium is a web automation framework for testing web applications."],
            output="Selenium is a web automation framework for testing.",
            metric="faithfulness"
        )
        
        evaluator = init_evaluator_from_env()
        score, explanation = evaluator.evaluate(
            metric_name="faithfulness",
            query=test_req.query,
            context=test_req.context,
            output=test_req.output,
            expected_output=None
        )
        
        return {
            "status": "ok",
            "test": "faithfulness_smoke_test",
            "example": {
                "query": test_req.query,
                "context": test_req.context,
                "output": test_req.output,
                "metric": test_req.metric
            },
            "result": {
                "score": score,
                "explanation": explanation,
                "expected_range": "0.8-1.0 (faithful output should score high)"
            }
        }
    except Exception as e:
        logger.exception("Smoke test failed")
        return {
            "status": "error",
            "test": "faithfulness_smoke_test",
            "error": str(e)
        }


@app.get("/metrics-info")
async def metrics_info():
    """Get available metrics information including required and recommended fields per metric.
    
    Provides complete field requirements:
    - faithfulness: output (required), context + query (recommended)
    - answer_relevancy: query + output (required)
    - contextual_precision/recall: context + output + expected_output (required)
    """
    metrics = []
    
    # Define required/optional fields per metric
    metric_requirements = {
        "faithfulness": {
            "required": ["output"],
            "recommended": ["context", "query"],
            "optional": []
        },
        "answer_relevancy": {
            "required": ["query", "output"],
            "recommended": ["context"],
            "optional": []
        },
        "contextual_precision": {
            "required": ["context", "output", "expected_output"],
            "recommended": ["query"],
            "optional": []
        },
        "contextual_recall": {
            "required": ["context", "output", "expected_output"],
            "recommended": ["query"],
            "optional": []
        },
        "conversation_completeness": {
            "required": ["messages"],
            "recommended": [],
            "optional": ["query"]
        },
        "hallucination": {
            "required": ["output", "context"],
            "recommended": ["query"],
            "optional": []
        }
    }
    
    for metric_name, description in MetricEvaluator.SUPPORTED_METRICS.items():
        requirements = metric_requirements.get(metric_name, {})
        
        # Hallucination has inverse scoring (lower is better)
        higher_is_better = True if metric_name != "hallucination" else False
        
        metrics.append({
            "name": metric_name,
            "description": description,
            "endpoint": "/eval",
            "parameter": f'"metric": "{metric_name}"',
            "range": "0.0 to 1.0",
            "higher_is_better": higher_is_better,
            "required_fields": requirements.get("required", []),
            "recommended_fields": requirements.get("recommended", []),
            "optional_fields": requirements.get("optional", [])
        })
    
    return {
        "available_metrics": metrics,
        "usage": "Include 'metric' parameter in POST /eval request body. Can be a string, array, or 'all'",
        "multi_metric_support": {
            "single": 'metric="faithfulness"',
            "multiple": 'metric=["faithfulness", "answer_relevancy", "hallucination"]',
            "all": 'metric="all" - evaluates all applicable metrics'
        },
        "training_note": "Each metric can be used independently for step-by-step learning. Uses hybrid approach: strict_mode=False for natural LLM judgment, plus custom post-processing to catch common failures. Faithfulness detects entity hallucinations. Answer Relevancy enforces definitional answers for 'What is...' questions. Hallucination metric uses lower-is-better scoring (0 = no hallucinations). OpenAI models provide naturally stricter base scoring than Groq models.",
        "request_structure": {
            "query": "Optional[str] - The user's question or input",
            "context": "Optional[List[str]] - List of retrieved documents or source passages",
            "output": "str - The model's generated response to evaluate (REQUIRED for most metrics)",
            "metric": "str | List[str] - Which metric(s) to use: single string, array of strings, or 'all' (default: faithfulness)",
            "expected_output": "Optional[str] - Reference answer (required for contextual_* metrics)",
            "messages": "Optional[List[dict]] - List of conversation turns with {role, content} (required for conversation_completeness)"
        },
        "example_requests": {
            "faithfulness": {
                "query": "What is Selenium?",
                "context": ["Selenium is a web automation framework for testing."],
                "output": "Selenium is used for web testing",
                "metric": "faithfulness"
            },
            "answer_relevancy": {
                "query": "Can you help me write Selenium code?",
                "output": "Yes, here is a basic example: driver.get('https://example.com')",
                "metric": "answer_relevancy"
            },
            "contextual_precision": {
                "query": "What is Selenium used for?",
                "context": ["Selenium is for web testing", "Python is a programming language"],
                "output": "Selenium is for web automation testing",
                "expected_output": "Selenium is used for web testing",
                "metric": "contextual_precision"
            },
            "contextual_recall": {
                "query": "What is Selenium used for?",
                "context": ["Selenium is a web automation framework", "It supports multiple browsers"],
                "output": "Selenium is for web automation and testing across browsers",
                "expected_output": "Selenium is used for web automation testing",
                "metric": "contextual_recall"
            },
            "conversation_completeness": {
                "messages": [
                    {"role": "user", "content": "What is Selenium?"},
                    {"role": "assistant", "content": "Selenium is a web automation framework."},
                    {"role": "user", "content": "What languages does it support?"},
                    {"role": "assistant", "content": "It supports Python, Java, JavaScript, C#, and Ruby."}
                ],
                "metric": "conversation_completeness"
            },
            "hallucination": {
                "query": "What is Selenium?",
                "context": ["Selenium is a web automation framework for testing web applications."],
                "output": "Selenium is a CRM platform developed by Salesforce.",
                "metric": "hallucination"
            },
            "multiple_metrics": {
                "query": "What is Selenium?",
                "context": ["Selenium is a web automation framework for testing."],
                "output": "Selenium is used for web testing",
                "metric": ["faithfulness", "answer_relevancy", "hallucination"]
            },
            "all_metrics": {
                "query": "What is Selenium?",
                "context": ["Selenium is a web automation framework for testing."],
                "output": "Selenium is used for web testing",
                "expected_output": "Selenium is a web automation framework",
                "metric": "all"
            }
        }
    }


if __name__ == "__main__":
    import uvicorn
    
    logger.info("Starting Deepeval Evaluation Service...")
    logger.info("API documentation available at http://localhost:8000/docs")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
