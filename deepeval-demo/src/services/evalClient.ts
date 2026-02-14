import axios from "axios";
import { ENV } from "../config/env.js";

export interface MetricResult {
  metric_name: string;
  score?: number;
  explanation?: string;
  error?: string;
}

export interface EvalResult {
  results: MetricResult[];  // New: array of metric results
  // Legacy fields for backward compatibility
  metric_name?: string;
  score?: number;
  explanation?: string;
  error?: string;
}

/**
 * Call DeepEval service to evaluate using specified metric.
 * 
 * The new API expects: { query?, context?, output, metric, expected_output? }
 * where:
 * - query: user's question
 * - context: array of retrieved documents/passages
 * - output: model's response
 * - metric: which metric to evaluate
 * - expected_output: reference answer (for contextual_* metrics)
 */
export async function evalWithMetric(
  contextOrQuery: string | string[],
  output: string,
  metric: string = "faithfulness",
  provider?: string
): Promise<EvalResult> {
  // Validate output
  if (typeof output !== "string" || output.trim() === "") {
    throw new Error("output must be a non-empty string");
  }

  // Build payload for the new API
  const payload: any = {
    output,
    metric,
  };

  // Handle context/query based on metric
  if (metric === "answer_relevancy") {
    // answer_relevancy requires query
    if (typeof contextOrQuery === "string") {
      payload.query = contextOrQuery;
    } else {
      throw new Error("answer_relevancy requires query as string");
    }
  } else if (metric === "faithfulness") {
    // faithfulness works best with context array
    if (Array.isArray(contextOrQuery)) {
      payload.context = contextOrQuery;
    } else if (typeof contextOrQuery === "string") {
      payload.context = [contextOrQuery];  // Convert string to array
    }
  } else if (metric === "contextual_precision" || metric === "contextual_recall") {
    // contextual_* metrics require context array
    if (Array.isArray(contextOrQuery)) {
      payload.context = contextOrQuery;
    } else if (typeof contextOrQuery === "string") {
      payload.context = [contextOrQuery];
    }
    // Note: expected_output should be provided separately via a different parameter
  }

  if (provider) {
    payload.provider = provider;
  }

  try {
    const res = await axios.post<EvalResult>(ENV.DEEPEVAL_URL, payload);
    return res.data;
  } catch (err: unknown) {
    if (axios.isAxiosError(err)) {
      if ((err as any).code === "ECONNREFUSED") {
        throw new Error(
          `DeepEval service unavailable at ${ENV.DEEPEVAL_URL}. Is it running?`
        );
      }
      const errorDetail = err.response?.data?.detail || err.message;
      throw new Error(
        `DeepEval Error (${err.response?.status || 'unknown'}): ${errorDetail}`
      );
    }
    throw err;
  }
}

/**
 * Evaluate with full control over all fields
 */
export async function evalWithFields(params: {
  query?: string;
  context?: string[];
  output?: string;
  metric?: string;
  expected_output?: string;
  provider?: string;
  messages?: Array<{ role: string; content: string }>;
}): Promise<EvalResult> {
  const payload: any = {
    metric: params.metric || "faithfulness",
  };

  // For conversation_completeness, output is optional but messages is required
  if (params.metric === "conversation_completeness") {
    if (!params.messages) {
      throw new Error("messages field is required for conversation_completeness metric");
    }
    payload.messages = params.messages;
    // output is still required by the API, provide a default if not given
    payload.output = params.output || "Conversation evaluation";
  } else {
    // For other metrics, output is required
    if (!params.output) {
      throw new Error("output field is required");
    }
    payload.output = params.output;
  }

  if (params.query) payload.query = params.query;
  if (params.context) payload.context = params.context;
  if (params.expected_output) payload.expected_output = params.expected_output;
  if (params.provider) payload.provider = params.provider;

  try {
    const res = await axios.post<EvalResult>(ENV.DEEPEVAL_URL, payload);
    return res.data;
  } catch (err: unknown) {
    if (axios.isAxiosError(err)) {
      if ((err as any).code === "ECONNREFUSED") {
        throw new Error(
          `DeepEval service unavailable at ${ENV.DEEPEVAL_URL}. Is it running?`
        );
      }
      const errorDetail = err.response?.data?.detail || err.message;
      throw new Error(
        `DeepEval Error (${err.response?.status || 'unknown'}): ${errorDetail}`
      );
    }
    throw err;
  }
}

/**
 * Legacy function for backward compatibility - defaults to faithfulness
 */
export async function evalFaithfulness(
  contextOrQuery: string | string[],
  output: string,
  provider?: string
): Promise<EvalResult> {
  return evalWithMetric(contextOrQuery, output, "faithfulness", provider);
}
