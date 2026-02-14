import express, { Request, Response, NextFunction } from "express";
import { ENV } from "./config/env.js";
import evalRoutes from "./routes/evalRoutes.js";

const app = express();

// Middleware
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Request logging
app.use((req: Request, res: Response, next: NextFunction) => {
  console.log(`[${new Date().toISOString()}] ${req.method} ${req.path}`);
  next();
});

// Routes
app.use("/api", evalRoutes);

// 404 handler
app.use((req: Request, res: Response) => {
  res.status(404).json({
    error: "Endpoint not found",
    path: req.path,
    method: req.method
  });
});

// Global error handler
app.use((err: unknown, req: Request, res: Response, next: NextFunction) => {
  console.error("[ERROR]", err);

  const error = err as Error & { status?: number };
  const status = error.status || 500;
  const message = error.message || "Internal Server Error";

  res.status(status).json({
    error: message,
    status
  });
});

// Start server
const PORT = ENV.PORT;
app.listen(PORT, () => {
  console.log(`
╔════════════════════════════════════════╗
║   Deepeval Demo Server Running         ║
╠════════════════════════════════════════╣
║   Port: ${PORT}
║   Env: ${process.env.NODE_ENV || "development"}
║   LLM Provider: groq
║   Deepeval URL: ${ENV.DEEPEVAL_URL}
╠════════════════════════════════════════╣
║   POST /api/llm/eval    (LLM + Eval)   ║
║   POST /api/rag/eval    (RAG + Eval)   ║
║   POST /api/eval-only   (Direct Eval)  ║
║   GET  /api/metrics     (Available)    ║
║   GET  /api/health      (Status)       ║
╚════════════════════════════════════════╝
  `);
});

export default app;
