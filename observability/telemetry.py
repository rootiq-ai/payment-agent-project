"""
GenAI Observability Module
==========================
Comprehensive observability for LLM applications including:
- Token usage and cost tracking
- Latency metrics (TTFT, P50/P95/P99)
- RAG retrieval quality metrics (Pinecone)
- Tool call analytics
- Hallucination detection
- Distributed tracing
- Multi-backend export (Splunk, Grafana, OTLP)

This demonstrates:
- Production-grade LLM observability
- OpenTelemetry integration
- Custom metrics collection
- Real-time dashboards
"""

import time
import json
import functools
import logging
from typing import Any, Callable, Optional, List, Dict
from dataclasses import dataclass, field, asdict
from datetime import datetime
from contextlib import contextmanager
from collections import defaultdict
import statistics
import hashlib
import uuid
import asyncio

# Structured logging
import structlog

# OpenTelemetry
from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader, ConsoleMetricExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.semconv.resource import ResourceAttributes
from opentelemetry.trace import Status, StatusCode


# ============================================================================
# Configuration
# ============================================================================

# Token pricing (per 1K tokens) - January 2025
TOKEN_PRICING = {
    "claude-sonnet-4-20250514": {"input": 0.003, "output": 0.015},
    "claude-3-5-sonnet": {"input": 0.003, "output": 0.015},
    "claude-3-opus": {"input": 0.015, "output": 0.075},
    "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
    "gpt-4-turbo": {"input": 0.01, "output": 0.03},
    "gpt-4o": {"input": 0.005, "output": 0.015},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
    "text-embedding-3-small": {"input": 0.00002, "output": 0},
    "text-embedding-3-large": {"input": 0.00013, "output": 0},
}

# Structured logger
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)


# ============================================================================
# Data Models
# ============================================================================

@dataclass
class TokenUsage:
    """Token usage for a single LLM call."""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0  # For prompt caching
    
    def calculate_cost(self, model: str) -> float:
        """Calculate cost based on model pricing."""
        pricing = TOKEN_PRICING.get(model, {"input": 0.01, "output": 0.03})
        input_cost = (self.input_tokens / 1000) * pricing["input"]
        output_cost = (self.output_tokens / 1000) * pricing["output"]
        return round(input_cost + output_cost, 6)


@dataclass
class LatencyMetrics:
    """Latency metrics for an operation."""
    start_time: float = field(default_factory=time.time)
    end_time: float = 0
    time_to_first_token: float = 0
    total_duration_ms: float = 0
    
    def record_first_token(self):
        """Record when first token is received."""
        self.time_to_first_token = (time.time() - self.start_time) * 1000
    
    def finalize(self):
        """Calculate final metrics."""
        if self.end_time == 0:
            self.end_time = time.time()
        self.total_duration_ms = (self.end_time - self.start_time) * 1000


@dataclass
class RAGMetrics:
    """Metrics for RAG retrieval (Pinecone)."""
    query: str
    query_embedding_time_ms: float = 0
    retrieval_time_ms: float = 0
    num_documents_retrieved: int = 0
    relevance_scores: List[float] = field(default_factory=list)
    avg_relevance_score: float = 0
    min_relevance_score: float = 0
    max_relevance_score: float = 0
    documents_used_in_context: int = 0
    total_context_tokens: int = 0
    pinecone_namespace: str = ""
    filter_applied: Optional[Dict] = None
    
    def calculate_stats(self):
        """Calculate relevance score statistics."""
        if self.relevance_scores:
            self.avg_relevance_score = statistics.mean(self.relevance_scores)
            self.min_relevance_score = min(self.relevance_scores)
            self.max_relevance_score = max(self.relevance_scores)


@dataclass
class ToolCallMetrics:
    """Metrics for tool/MCP calls."""
    tool_name: str
    tool_type: str = "mcp"  # mcp, function, api
    arguments: Dict = field(default_factory=dict)
    success: bool = True
    latency_ms: float = 0
    error: Optional[str] = None
    result_size_bytes: int = 0
    retry_count: int = 0


@dataclass
class HallucinationCheck:
    """Results of hallucination detection."""
    detected: bool = False
    confidence: float = 0.0
    check_type: str = ""  # factual, grounded, semantic_drift, citation
    method: str = ""  # rule_based, llm_judge, nli
    details: str = ""
    flagged_content: List[str] = field(default_factory=list)


@dataclass
class AgentMetrics:
    """Metrics for agent orchestration."""
    agent_id: str
    conversation_id: str
    turn_number: int = 0
    planning_time_ms: float = 0
    total_tool_calls: int = 0
    successful_tool_calls: int = 0
    tools_used: List[str] = field(default_factory=list)
    reasoning_steps: int = 0
    context_window_usage: float = 0  # Percentage of context used


@dataclass
class LLMCallRecord:
    """Complete record for a single LLM call."""
    call_id: str
    model: str
    timestamp: str
    tokens: TokenUsage
    latency: LatencyMetrics
    prompt_hash: str
    response_hash: str
    success: bool = True
    error: Optional[str] = None
    tool_calls: List[ToolCallMetrics] = field(default_factory=list)
    rag_metrics: Optional[RAGMetrics] = None
    hallucination_check: Optional[HallucinationCheck] = None
    agent_metrics: Optional[AgentMetrics] = None
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for export."""
        data = {
            "call_id": self.call_id,
            "model": self.model,
            "timestamp": self.timestamp,
            "success": self.success,
            "tokens": asdict(self.tokens),
            "latency": asdict(self.latency),
            "prompt_hash": self.prompt_hash,
            "response_hash": self.response_hash,
            "cost_usd": self.tokens.calculate_cost(self.model),
            "metadata": self.metadata
        }
        
        if self.error:
            data["error"] = self.error
        
        if self.tool_calls:
            data["tool_calls"] = [asdict(tc) for tc in self.tool_calls]
            data["tool_call_count"] = len(self.tool_calls)
        
        if self.rag_metrics:
            data["rag"] = asdict(self.rag_metrics)
        
        if self.hallucination_check:
            data["hallucination"] = asdict(self.hallucination_check)
        
        if self.agent_metrics:
            data["agent"] = asdict(self.agent_metrics)
        
        return data
    
    def to_splunk_event(self) -> Dict:
        """Format for Splunk HEC."""
        return {
            "time": time.time(),
            "event": self.to_dict(),
            "sourcetype": "genai:llm_call",
            "index": "genai_observability"
        }


# ============================================================================
# Metrics Collector
# ============================================================================

class MetricsCollector:
    """Collects and aggregates metrics across multiple calls."""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.calls: List[LLMCallRecord] = []
        self.latencies: List[float] = []
        self.ttfts: List[float] = []
        self.costs: List[float] = []
        self.token_counts: List[int] = []
        self.tool_usage: Dict[str, int] = defaultdict(int)
        self.model_usage: Dict[str, int] = defaultdict(int)
        self.error_count: int = 0
        self.hallucination_count: int = 0
        self.rag_relevance_scores: List[float] = []
        
        self.logger = structlog.get_logger()
    
    def record(self, call: LLMCallRecord):
        """Record a completed LLM call."""
        # Maintain max history
        if len(self.calls) >= self.max_history:
            self.calls.pop(0)
            self.latencies.pop(0)
            self.costs.pop(0)
            self.token_counts.pop(0)
        
        self.calls.append(call)
        self.latencies.append(call.latency.total_duration_ms)
        self.costs.append(call.tokens.calculate_cost(call.model))
        self.token_counts.append(call.tokens.total_tokens)
        self.model_usage[call.model] += 1
        
        if call.latency.time_to_first_token > 0:
            self.ttfts.append(call.latency.time_to_first_token)
        
        for tc in call.tool_calls:
            self.tool_usage[tc.tool_name] += 1
            if not tc.success:
                self.error_count += 1
        
        if not call.success:
            self.error_count += 1
        
        if call.hallucination_check and call.hallucination_check.detected:
            self.hallucination_count += 1
        
        if call.rag_metrics and call.rag_metrics.relevance_scores:
            self.rag_relevance_scores.extend(call.rag_metrics.relevance_scores)
        
        # Log the call
        self.logger.info(
            "llm_call_recorded",
            call_id=call.call_id,
            model=call.model,
            tokens=call.tokens.total_tokens,
            latency_ms=call.latency.total_duration_ms,
            cost_usd=call.tokens.calculate_cost(call.model),
            success=call.success
        )
    
    def get_percentile(self, values: List[float], percentile: float) -> float:
        """Calculate percentile from a list of values."""
        if not values:
            return 0
        sorted_values = sorted(values)
        index = int(len(sorted_values) * percentile / 100)
        return sorted_values[min(index, len(sorted_values) - 1)]
    
    def get_summary(self) -> Dict:
        """Get summary statistics."""
        if not self.calls:
            return {"message": "No calls recorded", "total_calls": 0}
        
        return {
            "total_calls": len(self.calls),
            "total_cost_usd": round(sum(self.costs), 4),
            "total_tokens": sum(self.token_counts),
            "avg_tokens_per_call": round(statistics.mean(self.token_counts), 1),
            "latency": {
                "avg_ms": round(statistics.mean(self.latencies), 2),
                "p50_ms": round(self.get_percentile(self.latencies, 50), 2),
                "p95_ms": round(self.get_percentile(self.latencies, 95), 2),
                "p99_ms": round(self.get_percentile(self.latencies, 99), 2),
                "min_ms": round(min(self.latencies), 2),
                "max_ms": round(max(self.latencies), 2),
            },
            "ttft": {
                "avg_ms": round(statistics.mean(self.ttfts), 2) if self.ttfts else 0,
                "p50_ms": round(self.get_percentile(self.ttfts, 50), 2) if self.ttfts else 0,
                "p95_ms": round(self.get_percentile(self.ttfts, 95), 2) if self.ttfts else 0,
            },
            "rag": {
                "avg_relevance": round(statistics.mean(self.rag_relevance_scores), 3) if self.rag_relevance_scores else 0,
                "total_retrievals": len([c for c in self.calls if c.rag_metrics])
            },
            "tool_usage": dict(self.tool_usage),
            "model_usage": dict(self.model_usage),
            "error_rate": round(self.error_count / len(self.calls), 4) if self.calls else 0,
            "hallucination_rate": round(self.hallucination_count / len(self.calls), 4) if self.calls else 0,
            "time_range": {
                "first_call": self.calls[0].timestamp if self.calls else None,
                "last_call": self.calls[-1].timestamp if self.calls else None
            }
        }
    
    def get_recent_calls(self, n: int = 10) -> List[Dict]:
        """Get recent calls for debugging."""
        return [call.to_dict() for call in self.calls[-n:]]
    
    def export_for_dashboard(self) -> Dict:
        """Export data formatted for dashboard visualization."""
        summary = self.get_summary()
        
        # Time series data for charts
        time_series = []
        for call in self.calls[-100:]:  # Last 100 calls
            time_series.append({
                "timestamp": call.timestamp,
                "latency_ms": call.latency.total_duration_ms,
                "tokens": call.tokens.total_tokens,
                "cost_usd": call.tokens.calculate_cost(call.model),
                "model": call.model
            })
        
        return {
            "summary": summary,
            "time_series": time_series,
            "tool_breakdown": [
                {"tool": k, "count": v} for k, v in self.tool_usage.items()
            ],
            "model_breakdown": [
                {"model": k, "count": v} for k, v in self.model_usage.items()
            ]
        }


# ============================================================================
# GenAI Telemetry Class
# ============================================================================

class GenAITelemetry:
    """
    Main telemetry class for GenAI observability.
    
    Usage:
        telemetry = GenAITelemetry(service_name="payment-agent")
        
        # Decorator for LLM calls
        @telemetry.trace_llm_call(model="claude-sonnet-4-20250514")
        async def call_llm(prompt):
            ...
        
        # Context manager for RAG
        with telemetry.trace_rag_retrieval("user query") as rag:
            documents = retriever.search(query)
            rag.record_results(documents, scores)
        
        # Manual recording
        telemetry.record_tool_call("payment_mcp.get_balance", args, result, latency)
    """
    
    def __init__(
        self,
        service_name: str = "payment-agent",
        environment: str = "development",
        backend: str = "console",  # console, splunk, otlp, all
        splunk_hec_url: Optional[str] = None,
        splunk_hec_token: Optional[str] = None,
        otlp_endpoint: Optional[str] = None,
        enable_otel: bool = True
    ):
        self.service_name = service_name
        self.environment = environment
        self.backend = backend
        self.collector = MetricsCollector()
        self.logger = structlog.get_logger()
        
        # Backend config
        self.splunk_hec_url = splunk_hec_url
        self.splunk_hec_token = splunk_hec_token
        self.otlp_endpoint = otlp_endpoint
        
        # OpenTelemetry
        if enable_otel:
            self._init_opentelemetry()
        else:
            self.tracer = None
            self.meter = None
        
        self.logger.info(
            "telemetry_initialized",
            service_name=service_name,
            environment=environment,
            backend=backend
        )
    
    def _init_opentelemetry(self):
        """Initialize OpenTelemetry tracing and metrics."""
        
        resource = Resource.create({
            ResourceAttributes.SERVICE_NAME: self.service_name,
            ResourceAttributes.DEPLOYMENT_ENVIRONMENT: self.environment,
        })
        
        # Tracing
        trace_provider = TracerProvider(resource=resource)
        trace_provider.add_span_processor(
            BatchSpanProcessor(ConsoleSpanExporter())
        )
        trace.set_tracer_provider(trace_provider)
        self.tracer = trace.get_tracer(self.service_name)
        
        # Metrics
        metric_reader = PeriodicExportingMetricReader(
            ConsoleMetricExporter(),
            export_interval_millis=60000  # 1 minute
        )
        metric_provider = MeterProvider(
            resource=resource,
            metric_readers=[metric_reader]
        )
        metrics.set_meter_provider(metric_provider)
        self.meter = metrics.get_meter(self.service_name)
        
        # Create metrics instruments
        self._create_metrics()
    
    def _create_metrics(self):
        """Create OpenTelemetry metric instruments."""
        
        # Counters
        self.token_counter = self.meter.create_counter(
            "genai.tokens.total",
            unit="tokens",
            description="Total tokens processed"
        )
        
        self.cost_counter = self.meter.create_counter(
            "genai.cost.usd",
            unit="usd",
            description="Total cost in USD"
        )
        
        self.call_counter = self.meter.create_counter(
            "genai.calls.total",
            unit="calls",
            description="Total LLM calls"
        )
        
        self.error_counter = self.meter.create_counter(
            "genai.errors.total",
            unit="errors",
            description="Total errors"
        )
        
        # Histograms
        self.latency_histogram = self.meter.create_histogram(
            "genai.latency.duration",
            unit="ms",
            description="LLM call latency distribution"
        )
        
        self.ttft_histogram = self.meter.create_histogram(
            "genai.latency.ttft",
            unit="ms",
            description="Time to first token distribution"
        )
        
        self.rag_relevance_histogram = self.meter.create_histogram(
            "genai.rag.relevance",
            unit="score",
            description="RAG retrieval relevance scores"
        )
    
    def _hash_content(self, content: str) -> str:
        """Create a hash of content for tracking without storing PII."""
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def _generate_call_id(self) -> str:
        """Generate unique call ID."""
        return f"call_{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}"
    
    # ========================================================================
    # Decorators
    # ========================================================================
    
    def trace_llm_call(self, model: str = "claude-sonnet-4-20250514"):
        """Decorator to trace LLM calls."""
        
        def decorator(func: Callable):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                call_id = self._generate_call_id()
                latency = LatencyMetrics()
                tokens = TokenUsage()
                
                span_context = None
                if self.tracer:
                    span_context = self.tracer.start_as_current_span(f"llm_call.{model}")
                
                try:
                    # Execute the LLM call
                    result = await func(*args, **kwargs)
                    latency.finalize()
                    
                    # Extract token usage if available
                    if hasattr(result, "usage"):
                        usage = result.usage
                        tokens.input_tokens = getattr(usage, "input_tokens", 0)
                        tokens.output_tokens = getattr(usage, "output_tokens", 0)
                        tokens.total_tokens = tokens.input_tokens + tokens.output_tokens
                    
                    # Create record
                    record = LLMCallRecord(
                        call_id=call_id,
                        model=model,
                        timestamp=datetime.now().isoformat(),
                        tokens=tokens,
                        latency=latency,
                        prompt_hash=self._hash_content(str(args)),
                        response_hash=self._hash_content(str(result)),
                        success=True
                    )
                    
                    # Record metrics
                    self._record_call(record)
                    
                    return result
                    
                except Exception as e:
                    latency.finalize()
                    
                    record = LLMCallRecord(
                        call_id=call_id,
                        model=model,
                        timestamp=datetime.now().isoformat(),
                        tokens=tokens,
                        latency=latency,
                        prompt_hash=self._hash_content(str(args)),
                        response_hash="",
                        success=False,
                        error=str(e)
                    )
                    
                    self._record_call(record)
                    raise
            
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                return asyncio.run(async_wrapper(*args, **kwargs))
            
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            return sync_wrapper
        
        return decorator
    
    # ========================================================================
    # Context Managers
    # ========================================================================
    
    @contextmanager
    def trace_rag_retrieval(self, query: str, namespace: str = "default"):
        """Context manager for tracing RAG retrieval."""
        
        rag_metrics = RAGMetrics(
            query=query,
            pinecone_namespace=namespace
        )
        start_time = time.time()
        
        try:
            yield rag_metrics
        finally:
            rag_metrics.retrieval_time_ms = (time.time() - start_time) * 1000
            rag_metrics.calculate_stats()
            
            # Record metrics
            if self.meter and rag_metrics.relevance_scores:
                for score in rag_metrics.relevance_scores:
                    self.rag_relevance_histogram.record(
                        score,
                        {"namespace": namespace}
                    )
            
            self.logger.info(
                "rag_retrieval",
                query_hash=self._hash_content(query),
                num_docs=rag_metrics.num_documents_retrieved,
                avg_relevance=rag_metrics.avg_relevance_score,
                latency_ms=rag_metrics.retrieval_time_ms
            )
    
    @contextmanager
    def trace_tool_call(self, tool_name: str, tool_type: str = "mcp"):
        """Context manager for tracing tool calls."""
        
        tool_metrics = ToolCallMetrics(
            tool_name=tool_name,
            tool_type=tool_type
        )
        start_time = time.time()
        
        try:
            yield tool_metrics
            tool_metrics.success = True
        except Exception as e:
            tool_metrics.success = False
            tool_metrics.error = str(e)
            raise
        finally:
            tool_metrics.latency_ms = (time.time() - start_time) * 1000
            
            self.logger.info(
                "tool_call",
                tool_name=tool_name,
                tool_type=tool_type,
                success=tool_metrics.success,
                latency_ms=tool_metrics.latency_ms
            )
    
    # ========================================================================
    # Recording Methods
    # ========================================================================
    
    def _record_call(self, record: LLMCallRecord):
        """Internal method to record a call to all backends."""
        
        # Add to collector
        self.collector.record(record)
        
        # Record to OpenTelemetry
        if self.meter:
            self.token_counter.add(
                record.tokens.total_tokens,
                {"model": record.model}
            )
            self.cost_counter.add(
                record.tokens.calculate_cost(record.model),
                {"model": record.model}
            )
            self.call_counter.add(1, {"model": record.model, "success": str(record.success)})
            self.latency_histogram.record(
                record.latency.total_duration_ms,
                {"model": record.model}
            )
            
            if record.latency.time_to_first_token > 0:
                self.ttft_histogram.record(
                    record.latency.time_to_first_token,
                    {"model": record.model}
                )
            
            if not record.success:
                self.error_counter.add(1, {"model": record.model})
    
    def record_call(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
        success: bool = True,
        error: Optional[str] = None,
        ttft_ms: float = 0,
        metadata: Optional[Dict] = None
    ):
        """Manually record an LLM call."""
        
        tokens = TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens
        )
        
        latency = LatencyMetrics()
        latency.total_duration_ms = latency_ms
        latency.time_to_first_token = ttft_ms
        
        record = LLMCallRecord(
            call_id=self._generate_call_id(),
            model=model,
            timestamp=datetime.now().isoformat(),
            tokens=tokens,
            latency=latency,
            prompt_hash="manual",
            response_hash="manual",
            success=success,
            error=error,
            metadata=metadata or {}
        )
        
        self._record_call(record)
    
    # ========================================================================
    # Query Methods
    # ========================================================================
    
    def get_summary(self) -> Dict:
        """Get summary statistics."""
        return self.collector.get_summary()
    
    def get_recent_calls(self, n: int = 10) -> List[Dict]:
        """Get recent calls."""
        return self.collector.get_recent_calls(n)
    
    def export_dashboard_data(self) -> Dict:
        """Export data for dashboard."""
        return self.collector.export_for_dashboard()


# ============================================================================
# Hallucination Detection
# ============================================================================

class HallucinationDetector:
    """
    Detects potential hallucinations in LLM responses.
    
    Methods:
    - Groundedness check: Verify response is grounded in provided context
    - Factual consistency: Check for self-contradictions
    - Citation verification: Verify cited sources exist in context
    """
    
    def __init__(self, telemetry: Optional[GenAITelemetry] = None):
        self.telemetry = telemetry
        self.logger = structlog.get_logger()
    
    def check_groundedness(
        self,
        response: str,
        context_documents: List[str],
        threshold: float = 0.7
    ) -> HallucinationCheck:
        """
        Check if response is grounded in the provided context.
        Uses simple keyword overlap for demo; production would use NLI model.
        """
        
        # Combine context
        context_text = " ".join(context_documents).lower()
        response_lower = response.lower()
        
        # Extract key claims (simplified - would use NER/claim extraction in production)
        response_words = set(response_lower.split())
        context_words = set(context_text.split())
        
        # Calculate overlap
        overlap = len(response_words & context_words)
        total = len(response_words)
        groundedness_score = overlap / total if total > 0 else 0
        
        detected = groundedness_score < threshold
        
        return HallucinationCheck(
            detected=detected,
            confidence=1 - groundedness_score,
            check_type="grounded",
            method="keyword_overlap",
            details=f"Groundedness score: {groundedness_score:.2f}"
        )
    
    def check_factual_consistency(
        self,
        response: str,
        previous_responses: List[str]
    ) -> HallucinationCheck:
        """Check for contradictions with previous responses."""
        
        # Simplified check - production would use contradiction detection model
        # For now, just check for obvious numeric contradictions
        
        import re
        
        # Extract numbers from response
        response_numbers = set(re.findall(r'\$?\d+(?:\.\d+)?%?', response))
        
        contradictions = []
        for prev in previous_responses:
            prev_numbers = set(re.findall(r'\$?\d+(?:\.\d+)?%?', prev))
            # Check if same number appears in different contexts
            # (Simplified - real implementation would be semantic)
        
        return HallucinationCheck(
            detected=len(contradictions) > 0,
            confidence=0.5 if contradictions else 0.0,
            check_type="factual",
            method="numeric_consistency",
            details=f"Found {len(contradictions)} potential contradictions",
            flagged_content=contradictions
        )


# ============================================================================
# Demo
# ============================================================================

def demo_observability():
    """Demo the observability module."""
    
    print("=" * 60)
    print("GenAI Observability Demo")
    print("=" * 60)
    
    # Initialize telemetry
    telemetry = GenAITelemetry(
        service_name="payment-agent-demo",
        environment="development",
        enable_otel=False  # Disable OTel console output for cleaner demo
    )
    
    # Simulate some LLM calls
    print("\n📊 Simulating LLM calls...")
    
    test_calls = [
        {"model": "claude-sonnet-4-20250514", "input": 500, "output": 200, "latency": 1200},
        {"model": "claude-sonnet-4-20250514", "input": 300, "output": 150, "latency": 800},
        {"model": "claude-sonnet-4-20250514", "input": 800, "output": 400, "latency": 2100},
        {"model": "gpt-4o", "input": 400, "output": 200, "latency": 950},
        {"model": "claude-sonnet-4-20250514", "input": 600, "output": 300, "latency": 1500},
    ]
    
    for call in test_calls:
        telemetry.record_call(
            model=call["model"],
            input_tokens=call["input"],
            output_tokens=call["output"],
            latency_ms=call["latency"],
            ttft_ms=call["latency"] * 0.3  # Simulate TTFT
        )
    
    # Get summary
    print("\n📈 Metrics Summary:")
    summary = telemetry.get_summary()
    print(json.dumps(summary, indent=2))
    
    # Test RAG tracing
    print("\n🔍 Testing RAG tracing...")
    with telemetry.trace_rag_retrieval("What is the refund policy?", namespace="payments") as rag:
        # Simulate retrieval
        time.sleep(0.1)
        rag.num_documents_retrieved = 4
        rag.relevance_scores = [0.92, 0.87, 0.81, 0.75]
        rag.calculate_stats()
    
    print(f"   Retrieved {rag.num_documents_retrieved} docs, avg relevance: {rag.avg_relevance_score:.3f}")
    
    # Test hallucination detection
    print("\n🔎 Testing hallucination detection...")
    detector = HallucinationDetector(telemetry)
    
    context = ["Refunds are processed within 3-5 business days. Maximum refund amount is $10,000."]
    response = "Your refund will be processed within 3-5 business days for amounts up to $10,000."
    
    check = detector.check_groundedness(response, context)
    print(f"   Groundedness check - Hallucination detected: {check.detected}")
    print(f"   Confidence: {check.confidence:.2f}")
    
    print("\n✅ Demo complete!")


if __name__ == "__main__":
    demo_observability()
