"""
Monitoring and Logging for Production
Structured logging, metrics, and alerts
"""

import logging
import json
import time
from datetime import datetime, timezone
from functools import wraps
from typing import Any, Callable
from langchain_openai import ChatOpenAI
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage
from langsmith import traceable
from dotenv import load_dotenv

load_dotenv()


# === Structured Logging ===


class JSONFormatter(logging.Formatter):
    """Format logs as JSON for log aggregation."""

    def format(self, record):
        log_obj = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
        }

        if hasattr(record, "extra_data"):
            log_obj.update(record.extra_data)

        return json.dumps(log_obj)


def setup_logging():
    """Setup structured JSON logging."""

    logger = logging.getLogger("langgraph_app")
    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler()
    handler.setFormatter(JSONFormatter())
    logger.addHandler(handler)

    return logger


# === Metrics Collection ===


class MetricsCollector:
    """Collect and aggregate metrics."""

    def __init__(self):
        self.metrics = {
            "requests_total": 0,
            "errors_total": 0,
            "latency_sum": 0,
            "latency_count": 0,
            "tokens_input": 0,
            "tokens_output": 0,
            "cache_hits": 0,
            "cache_misses": 0,
        }

    def record_request(
        self,
        latency_ms: float,
        input_tokens: int,
        output_tokens: int,
        error: bool = False,
        cache_hit: bool = False,
    ):
        self.metrics["requests_total"] += 1
        self.metrics["latency_sum"] += latency_ms
        self.metrics["latency_count"] += 1
        self.metrics["tokens_input"] += input_tokens
        self.metrics["tokens_output"] += output_tokens

        if error:
            self.metrics["errors_total"] += 1

        if cache_hit:
            self.metrics["cache_hits"] += 1
        else:
            self.metrics["cache_misses"] += 1

    def get_summary(self) -> dict:
        avg_latency = (
            self.metrics["latency_sum"] / self.metrics["latency_count"]
            if self.metrics["latency_count"] > 0
            else 0
        )
        error_rate = (
            self.metrics["errors_total"] / self.metrics["requests_total"]
            if self.metrics["requests_total"] > 0
            else 0
        )
        cache_hit_rate = (
            self.metrics["cache_hits"]
            / (self.metrics["cache_hits"] + self.metrics["cache_misses"])
            if (self.metrics["cache_hits"] + self.metrics["cache_misses"]) > 0
            else 0
        )

        return {
            "total_requests": self.metrics["requests_total"],
            "total_errors": self.metrics["errors_total"],
            "error_rate": f"{error_rate:.2%}",
            "avg_latency_ms": round(avg_latency, 2),
            "total_input_tokens": self.metrics["tokens_input"],
            "total_output_tokens": self.metrics["tokens_output"],
            "cache_hit_rate": f"{cache_hit_rate:.2%}",
        }


# === Instrumented LLM ===


class InstrumentedLLM:
    """LLM with full instrumentation."""

    def __init__(self):
        # NOTE (2026-11): gpt-4o-mini/gpt-4o shut down 2027-01-14 -> gpt-6-luna. reasoning_effort="none" lets it accept temperature + tools.
        self.llm = ChatOpenAI(model="gpt-6-luna", reasoning_effort="none", temperature=0)
        self.metrics = MetricsCollector()
        self.logger = setup_logging()

    @traceable(name="instrumented_invoke")
    def invoke(self, query: str) -> str:
        start_time = time.time()
        error = False

        try:
            response = self.llm.invoke(query)
            result = response.content

            # Estimate tokens
            input_tokens = len(query.split()) * 4 // 3
            output_tokens = len(result.split()) * 4 // 3

            self.metrics.record_request(
                latency_ms=(time.time() - start_time) * 1000,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                error=False,
                cache_hit=False,
            )

            self.logger.info(
                "LLM request completed",
                extra={
                    "extra_data": {
                        "latency_ms": (time.time() - start_time) * 1000,
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                    }
                },
            )

            return result

        except Exception as e:
            error = True
            self.metrics.record_request(
                latency_ms=(time.time() - start_time) * 1000,
                input_tokens=0,
                output_tokens=0,
                error=True,
                cache_hit=False,
            )

            self.logger.error(
                f"LLM request failed: {e}", extra={"extra_data": {"error": str(e)}}
            )

            raise


def demo_monitoring():
    """Demonstrate monitoring."""

    llm = InstrumentedLLM()

    print("Monitoring Demo:\n")

    queries = [
        "What is Python?",
        "Explain machine learning.",
        "What is 2 + 2?",
    ]

    for query in queries:
        result = llm.invoke(query)
        print(f"Query: {query[:30]}... -> {result[:30]}...")

    print("\nMetrics Summary:")
    summary = llm.metrics.get_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    # logger = setup_logging()
    # logger.info("Logging setup complete", extra={"extra_data": {"app": "langgraph"}})
    demo_monitoring()
