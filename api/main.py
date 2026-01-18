"""
Payment Agent API
=================
FastAPI server that exposes the AI payment agent via REST API.
Integrates MCP servers, RAG pipeline, and observability.

Endpoints:
- POST /chat - Chat with the payment agent
- GET /health - Health check
- GET /metrics - Observability metrics
- POST /rag/search - Direct RAG search
- GET /customer/{id} - Get customer info
"""

import os
import time
from typing import Optional, List, Dict, Any
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

import structlog

# Import our modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.orchestrator import PaymentAgent
from rag.pipeline import PaymentRAG, PaymentRAGChain
from observability.telemetry import GenAITelemetry


# ============================================================================
# Configuration
# ============================================================================

logger = structlog.get_logger()

# Environment
ENV = os.getenv("ENVIRONMENT", "development")
DEBUG = ENV == "development"

# API Keys check
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")


# ============================================================================
# Request/Response Models
# ============================================================================

class ChatRequest(BaseModel):
    """Chat request payload."""
    message: str = Field(..., min_length=1, max_length=10000)
    customer_id: str = Field(default="cust_demo")
    conversation_id: Optional[str] = None
    include_debug: bool = False


class ChatResponse(BaseModel):
    """Chat response payload."""
    response: str
    conversation_id: str
    customer_id: str
    timestamp: str
    debug: Optional[Dict] = None


class RAGSearchRequest(BaseModel):
    """RAG search request."""
    query: str = Field(..., min_length=1, max_length=1000)
    k: int = Field(default=4, ge=1, le=10)
    topic_filter: Optional[str] = None


class RAGSearchResponse(BaseModel):
    """RAG search response."""
    query: str
    results: List[Dict]
    count: int
    latency_ms: float


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    environment: str
    timestamp: str
    services: Dict[str, bool]


class MetricsResponse(BaseModel):
    """Metrics response."""
    summary: Dict
    recent_calls: List[Dict]


# ============================================================================
# Application State
# ============================================================================

class AppState:
    """Application state container."""
    
    def __init__(self):
        self.agent: Optional[PaymentAgent] = None
        self.rag: Optional[PaymentRAG] = None
        self.rag_chain: Optional[PaymentRAGChain] = None
        self.telemetry: Optional[GenAITelemetry] = None
        self.conversations: Dict[str, List] = {}
        self.startup_time: Optional[datetime] = None
    
    def initialize(self):
        """Initialize all components."""
        logger.info("initializing_app_state")
        
        self.startup_time = datetime.now()
        
        # Initialize telemetry first
        self.telemetry = GenAITelemetry(
            service_name="payment-agent-api",
            environment=ENV,
            enable_otel=True
        )
        
        # Initialize RAG (works with or without Pinecone)
        try:
            self.rag = PaymentRAG()
            self.rag_chain = PaymentRAGChain(self.rag)
            logger.info("rag_initialized", has_pinecone=bool(PINECONE_API_KEY))
        except Exception as e:
            logger.warning("rag_init_failed", error=str(e))
            self.rag = None
            self.rag_chain = None
        
        # Initialize agent
        try:
            self.agent = PaymentAgent()
            logger.info("agent_initialized")
        except Exception as e:
            logger.warning("agent_init_failed", error=str(e))
            self.agent = None
        
        logger.info("app_state_initialized")
    
    def get_conversation_history(self, conversation_id: str) -> List:
        """Get or create conversation history."""
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = []
        return self.conversations[conversation_id]


# Global state
state = AppState()


# ============================================================================
# Lifespan Management
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("starting_application")
    state.initialize()
    yield
    # Shutdown
    logger.info("shutting_down_application")


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="PaymentGPT API",
    description="AI-powered payment assistant with MCP, RAG, and observability",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Endpoints
# ============================================================================

@app.get("/", response_model=Dict)
async def root():
    """Root endpoint with API info."""
    return {
        "name": "PaymentGPT API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    
    services = {
        "agent": state.agent is not None,
        "rag": state.rag is not None,
        "telemetry": state.telemetry is not None,
        "anthropic_api": bool(ANTHROPIC_API_KEY),
        "openai_api": bool(OPENAI_API_KEY),
        "pinecone_api": bool(PINECONE_API_KEY)
    }
    
    all_critical_healthy = services["agent"] or services["rag"]
    
    return HealthResponse(
        status="healthy" if all_critical_healthy else "degraded",
        environment=ENV,
        timestamp=datetime.now().isoformat(),
        services=services
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, background_tasks: BackgroundTasks):
    """
    Chat with the payment agent.
    
    The agent can:
    - Check balances and transactions
    - Process refunds
    - Answer policy questions (via RAG)
    - Detect duplicate charges
    - And more...
    """
    
    start_time = time.time()
    
    # Generate conversation ID if not provided
    conversation_id = request.conversation_id or f"conv_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    # Check if agent is available
    if not state.agent:
        # Fallback to RAG-only response
        if state.rag_chain:
            result = state.rag_chain.query(request.message)
            response_text = result["answer"]
        else:
            raise HTTPException(
                status_code=503,
                detail="Agent service unavailable. Please check API keys."
            )
    else:
        # Use full agent
        try:
            response_text = state.agent.chat(
                message=request.message,
                customer_id=request.customer_id,
                thread_id=conversation_id
            )
        except Exception as e:
            logger.error("agent_chat_error", error=str(e))
            raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")
    
    latency_ms = (time.time() - start_time) * 1000
    
    # Record telemetry
    if state.telemetry:
        state.telemetry.record_call(
            model="claude-sonnet-4-20250514",
            input_tokens=len(request.message.split()) * 2,  # Rough estimate
            output_tokens=len(response_text.split()) * 2,
            latency_ms=latency_ms,
            metadata={
                "customer_id": request.customer_id,
                "conversation_id": conversation_id
            }
        )
    
    # Build response
    response = ChatResponse(
        response=response_text,
        conversation_id=conversation_id,
        customer_id=request.customer_id,
        timestamp=datetime.now().isoformat()
    )
    
    if request.include_debug:
        response.debug = {
            "latency_ms": round(latency_ms, 2),
            "agent_available": state.agent is not None,
            "rag_available": state.rag is not None
        }
    
    return response


@app.post("/rag/search", response_model=RAGSearchResponse)
async def rag_search(request: RAGSearchRequest):
    """
    Direct RAG search endpoint.
    
    Useful for:
    - Testing RAG retrieval quality
    - Finding relevant policy documents
    - Debugging retrieval issues
    """
    
    if not state.rag:
        raise HTTPException(
            status_code=503,
            detail="RAG service unavailable"
        )
    
    start_time = time.time()
    
    # Perform search
    if request.topic_filter:
        results = state.rag.search_by_topic(
            query=request.query,
            topic=request.topic_filter,
            k=request.k
        )
    else:
        results = state.rag.search_with_scores(
            query=request.query,
            k=request.k
        )
    
    latency_ms = (time.time() - start_time) * 1000
    
    # Format results
    formatted_results = []
    for item in results:
        if isinstance(item, tuple):
            doc, score = item
            formatted_results.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "relevance_score": round(score, 4)
            })
        else:
            formatted_results.append({
                "content": item.page_content,
                "metadata": item.metadata,
                "relevance_score": None
            })
    
    # Record telemetry
    if state.telemetry:
        with state.telemetry.trace_rag_retrieval(request.query) as rag_metrics:
            rag_metrics.num_documents_retrieved = len(formatted_results)
            if formatted_results and formatted_results[0].get("relevance_score"):
                rag_metrics.relevance_scores = [
                    r["relevance_score"] for r in formatted_results 
                    if r.get("relevance_score")
                ]
    
    return RAGSearchResponse(
        query=request.query,
        results=formatted_results,
        count=len(formatted_results),
        latency_ms=round(latency_ms, 2)
    )


@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """
    Get observability metrics.
    
    Returns:
    - Summary statistics (tokens, costs, latencies)
    - Recent calls for debugging
    - Tool usage breakdown
    """
    
    if not state.telemetry:
        raise HTTPException(
            status_code=503,
            detail="Telemetry service unavailable"
        )
    
    return MetricsResponse(
        summary=state.telemetry.get_summary(),
        recent_calls=state.telemetry.get_recent_calls(n=10)
    )


@app.get("/metrics/dashboard")
async def get_dashboard_data():
    """Get data formatted for dashboard visualization."""
    
    if not state.telemetry:
        raise HTTPException(status_code=503, detail="Telemetry unavailable")
    
    return state.telemetry.export_dashboard_data()


@app.get("/customer/{customer_id}")
async def get_customer(customer_id: str):
    """
    Get customer information.
    
    This is a simplified endpoint that would typically
    call the customer MCP server.
    """
    
    # Mock customer data (would use MCP in production)
    customers = {
        "cust_demo": {
            "customer_id": "cust_demo",
            "name": "Sarah Johnson",
            "email": "demo@example.com",
            "tier": "premium",
            "status": "active",
            "balance": 10000.00,
            "financing": {
                "solar_system_kw": 8.5,
                "monthly_payment": 150.00,
                "remaining_balance": 18000.00
            }
        },
        "cust_001": {
            "customer_id": "cust_001",
            "name": "John Doe",
            "email": "john@example.com",
            "tier": "standard",
            "status": "active",
            "balance": 5000.00
        }
    }
    
    customer = customers.get(customer_id)
    if not customer:
        raise HTTPException(status_code=404, detail=f"Customer {customer_id} not found")
    
    return customer


@app.get("/rag/stats")
async def get_rag_stats():
    """Get Pinecone index statistics."""
    
    if not state.rag:
        raise HTTPException(status_code=503, detail="RAG service unavailable")
    
    return state.rag.get_index_stats()


# ============================================================================
# Error Handlers
# ============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    
    logger.error(
        "unhandled_exception",
        error=str(exc),
        path=request.url.path
    )
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc) if DEBUG else "An unexpected error occurred"
        }
    )


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=DEBUG,
        log_level="info"
    )
