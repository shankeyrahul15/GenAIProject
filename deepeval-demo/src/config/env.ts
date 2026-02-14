import dotenv from "dotenv";

dotenv.config();

export const ENV = {
  PORT: process.env.PORT ? parseInt(process.env.PORT, 10) : 3000,
  OPENAI_API_KEY: process.env.OPENAI_API_KEY || "",
  GROQ_API_KEY: process.env.GROQ_API_KEY || "",
  DEEPEVAL_URL: process.env.DEEPEVAL_URL || "http://localhost:8000/eval"
};

// Validate required environment variables
if (!ENV.GROQ_API_KEY) {
  console.warn(
    "Warning: GROQ_API_KEY is not set. LLM calls will fail."
  );
}
