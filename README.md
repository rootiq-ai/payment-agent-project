# PaymentGPT — AI Payment Agent with MCP + RAG + Observability

A production-ready AI payment assistant demonstrating enterprise-grade agentic workflows, MCP server integration, RAG pipelines, and comprehensive GenAI observability.

## 🎯 Project Goals

This project demonstrates proficiency in:
- **MCP Server Development** — Building and deploying Model Context Protocol servers
- **Agentic Workflows** — LLM-powered agents orchestrating tools and APIs
- **RAG Pipelines** — Retrieval-Augmented Generation for payment knowledge
- **AI Observability** — Full telemetry for LLM applications in production
- **Cloud Architecture** — AWS-based deployment with Bedrock integration

## 📁 Project Structure

```
payment-agent-project/
├── mcp_servers/           # MCP Server implementations
│   ├── payment_mcp.py     # Payment operations server
│   ├── customer_mcp.py    # Customer data server
│   └── compliance_mcp.py  # Risk & compliance server
├── agent/                 # Agent orchestration
│   ├── orchestrator.py    # Main agent logic
│   ├── tools.py           # MCP tool wrappers
│   └── memory.py          # Conversation memory
├── rag/                   # RAG pipeline
│   ├── indexer.py         # Document indexing
│   ├── retriever.py       # Semantic search
│   └── documents/         # Knowledge base docs
├── observability/         # Telemetry & monitoring
│   ├── telemetry.py       # GenAI telemetry SDK
│   ├── metrics.py         # Custom metrics
│   └── traces.py          # Distributed tracing
├── api/                   # REST API layer
│   ├── main.py            # FastAPI application
│   ├── routes.py          # API endpoints
│   └── models.py          # Pydantic schemas
├── dashboards/            # Monitoring dashboards
│   ├── splunk/            # Splunk dashboards
│   └── grafana/           # Grafana dashboards
├── tests/                 # Test suite
├── docs/                  # Documentation
├── docker-compose.yml     # Local development
└── requirements.txt       # Dependencies
```

## 🗓️ Implementation Timeline

### Week 1: Foundation + MCP Servers
- Day 1-2: Project setup, dependencies, local environment
- Day 3-4: Build Payment MCP Server
- Day 5-6: Build Customer & Compliance MCP Servers
- Day 7: Integration testing for all MCP servers

### Week 2: Agent + RAG
- Day 8-9: Agent orchestrator with LangChain/LangGraph
- Day 10-11: RAG pipeline with vector database
- Day 12-13: Connect agent to MCP servers + RAG
- Day 14: End-to-end workflow testing

### Week 3: Observability + Polish
- Day 15-16: Integrate genai-telemetry SDK
- Day 17-18: Build Splunk/Grafana dashboards
- Day 19-20: Create demo scenarios
- Day 21: Documentation, video recording, final polish

## 🚀 Quick Start

```bash
# Clone and setup
cd payment-agent-project
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Set environment variables
cp .env.example .env
# Edit .env with your API keys

# Start MCP servers
python -m mcp_servers.payment_mcp &
python -m mcp_servers.customer_mcp &
python -m mcp_servers.compliance_mcp &

# Start the API
uvicorn api.main:app --reload

# Run demo
python demo.py
```

## 🔑 Environment Variables

```bash
# LLM Provider
ANTHROPIC_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
AWS_BEDROCK_REGION=us-east-1

# Vector Database
PINECONE_API_KEY=your_key_here
PINECONE_ENVIRONMENT=your_env

# Observability
SPLUNK_HEC_TOKEN=your_token
SPLUNK_HEC_URL=https://your-splunk:8088

# Database (for demo)
DATABASE_URL=sqlite:///./payment_demo.db
```

## 📊 Demo Scenarios

| # | Scenario | Skills Demonstrated |
|---|----------|---------------------|
| 1 | "What's my balance?" | MCP tool calling, customer data retrieval |
| 2 | "Show my recent transactions" | Multi-tool orchestration |
| 3 | "I was charged twice, help!" | Agentic workflow, investigation, refund |
| 4 | "What's your refund policy?" | RAG retrieval |
| 5 | "Process $500 payment" | Compliance check → fraud detection → payment |
| 6 | "Why was my payment declined?" | Root cause analysis, multi-source reasoning |

## 📈 Observability Metrics

- Token usage & costs per conversation
- Latency: TTFT, P50/P95/P99
- Tool selection patterns
- RAG retrieval relevance scores
- Hallucination detection rates
- Agent decision traces

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| LLM | Claude (Anthropic) / AWS Bedrock |
| Agent Framework | LangChain + LangGraph |
| MCP | Official Python SDK |
| Vector DB | Chroma (local) / Pinecone (prod) |
| Backend | FastAPI |
| Observability | genai-telemetry + Splunk/Grafana |
| Database | SQLite (demo) / PostgreSQL (prod) |

## 📄 License

MIT License - Built for interview demonstration purposes.
