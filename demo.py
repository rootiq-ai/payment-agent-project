#!/usr/bin/env python3
"""
PaymentGPT Demo Script
======================
Demonstrates all components of the AI Payment Agent:
1. MCP Server tools
2. Agent orchestration
3. RAG retrieval (Pinecone)
4. GenAI Observability

Run this script to see the full system in action.
"""

import os
import sys
import json
import time
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown
from rich import print as rprint


console = Console()


def print_header(title: str):
    """Print a section header."""
    console.print()
    console.print(Panel(title, style="bold blue"))


def print_step(step: str, description: str):
    """Print a step in the demo."""
    console.print(f"\n[bold cyan]→ {step}[/bold cyan]: {description}")


def check_environment():
    """Check required environment variables."""
    print_header("🔧 Environment Check")
    
    env_vars = {
        "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY"),
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "PINECONE_API_KEY": os.getenv("PINECONE_API_KEY"),
    }
    
    table = Table(title="API Keys Status")
    table.add_column("Variable", style="cyan")
    table.add_column("Status", style="green")
    
    all_set = True
    for var, value in env_vars.items():
        if value:
            table.add_row(var, "✅ Set")
        else:
            table.add_row(var, "[red]❌ Not Set[/red]")
            all_set = False
    
    console.print(table)
    
    if not all_set:
        console.print("\n[yellow]⚠️  Some API keys are missing. Demo will run in mock mode.[/yellow]")
    
    return all_set


def demo_mcp_servers():
    """Demonstrate MCP server functionality."""
    print_header("🔌 MCP Servers Demo")
    
    print_step("1", "Loading MCP server tools...")
    
    # Import tool functions from orchestrator (they're simulated there)
    from agent.orchestrator import (
        get_balance,
        get_transactions,
        detect_duplicate_charges,
        process_refund,
        get_customer_profile,
        check_fraud
    )
    
    # Demo: Get balance
    print_step("2", "Calling get_balance tool...")
    result = get_balance.invoke({"customer_id": "cust_demo"})
    console.print(f"   Balance: [green]${result['balance']:,.2f}[/green]")
    
    # Demo: Get transactions
    print_step("3", "Calling get_transactions tool...")
    result = get_transactions.invoke({"customer_id": "cust_demo", "limit": 3})
    console.print(f"   Found [cyan]{result['count']}[/cyan] transactions:")
    for txn in result['transactions'][:3]:
        console.print(f"      • {txn['transaction_id']}: ${txn['amount']:,.2f} - {txn['description'][:30]}...")
    
    # Demo: Detect duplicates
    print_step("4", "Calling detect_duplicate_charges tool...")
    result = detect_duplicate_charges.invoke({"customer_id": "cust_demo", "hours": 24})
    if result['duplicates_found'] > 0:
        console.print(f"   [yellow]⚠️  Found {result['duplicates_found']} potential duplicate(s)![/yellow]")
        for dup in result['duplicates']:
            console.print(f"      • Amount: ${dup['amount']:,.2f}")
    else:
        console.print("   [green]✓ No duplicates found[/green]")
    
    # Demo: Fraud check
    print_step("5", "Calling check_fraud tool...")
    result = check_fraud.invoke({
        "transaction_id": "txn_001",
        "customer_id": "cust_demo",
        "amount": 500.00
    })
    fraud_check = result['fraud_check']
    console.print(f"   Fraud Score: [{'red' if fraud_check['fraud_score'] > 50 else 'green'}]{fraud_check['fraud_score']}[/]")
    console.print(f"   Recommendation: {fraud_check['recommendation']}")


def demo_rag_pipeline():
    """Demonstrate RAG pipeline with Pinecone."""
    print_header("🔍 RAG Pipeline Demo (Pinecone)")
    
    print_step("1", "Initializing RAG with Pinecone...")
    
    from rag.pipeline import PaymentRAG, PAYMENT_KNOWLEDGE_DOCS
    
    rag = PaymentRAG()
    
    # Show index stats
    stats = rag.get_index_stats()
    if "error" not in stats:
        console.print(f"   Pinecone index connected: [green]✓[/green]")
        console.print(f"   Total vectors: {stats.get('total_vector_count', 'N/A')}")
    else:
        console.print(f"   [yellow]Running in mock mode (no Pinecone)[/yellow]")
    
    # Demo searches
    test_queries = [
        "What is the refund policy?",
        "How do I set up autopay?",
        "What happens if I miss a payment?",
    ]
    
    print_step("2", "Testing semantic search...")
    
    for i, query in enumerate(test_queries, 1):
        console.print(f"\n   Query {i}: \"{query}\"")
        results = rag.search_with_scores(query, k=2)
        
        for doc, score in results:
            topic = doc.metadata.get('topic', 'unknown')
            preview = doc.page_content[:80].replace('\n', ' ').strip()
            console.print(f"      [{score:.3f}] {topic}: {preview}...")


def demo_observability():
    """Demonstrate GenAI observability."""
    print_header("📊 GenAI Observability Demo")
    
    print_step("1", "Initializing telemetry...")
    
    from observability.telemetry import GenAITelemetry, HallucinationDetector
    
    telemetry = GenAITelemetry(
        service_name="payment-agent-demo",
        environment="demo",
        enable_otel=False  # Disable for cleaner output
    )
    
    # Simulate some LLM calls
    print_step("2", "Recording simulated LLM calls...")
    
    test_calls = [
        {"model": "claude-sonnet-4-20250514", "input": 500, "output": 200, "latency": 1200, "ttft": 350},
        {"model": "claude-sonnet-4-20250514", "input": 300, "output": 150, "latency": 800, "ttft": 250},
        {"model": "claude-sonnet-4-20250514", "input": 800, "output": 400, "latency": 2100, "ttft": 500},
        {"model": "gpt-4o", "input": 400, "output": 200, "latency": 950, "ttft": 280},
        {"model": "claude-sonnet-4-20250514", "input": 600, "output": 300, "latency": 1500, "ttft": 400},
    ]
    
    for call in test_calls:
        telemetry.record_call(
            model=call["model"],
            input_tokens=call["input"],
            output_tokens=call["output"],
            latency_ms=call["latency"],
            ttft_ms=call["ttft"]
        )
    
    console.print(f"   Recorded [cyan]{len(test_calls)}[/cyan] calls")
    
    # Show summary
    print_step("3", "Metrics summary...")
    
    summary = telemetry.get_summary()
    
    table = Table(title="Observability Metrics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Total Calls", str(summary['total_calls']))
    table.add_row("Total Tokens", f"{summary['total_tokens']:,}")
    table.add_row("Total Cost", f"${summary['total_cost_usd']:.4f}")
    table.add_row("Avg Latency", f"{summary['latency']['avg_ms']:.0f}ms")
    table.add_row("P95 Latency", f"{summary['latency']['p95_ms']:.0f}ms")
    table.add_row("Avg TTFT", f"{summary['ttft']['avg_ms']:.0f}ms")
    
    console.print(table)
    
    # Hallucination detection
    print_step("4", "Testing hallucination detection...")
    
    detector = HallucinationDetector()
    
    context = ["Refunds are processed within 3-5 business days. Maximum refund is $10,000."]
    response = "Your refund will be processed within 3-5 business days."
    
    check = detector.check_groundedness(response, context)
    console.print(f"   Response: \"{response}\"")
    console.print(f"   Grounded: [{'green' if not check.detected else 'red'}]{'Yes' if not check.detected else 'No'}[/]")


def demo_agent_conversation():
    """Demonstrate full agent conversation."""
    print_header("🤖 Agent Conversation Demo")
    
    print_step("1", "Initializing agent...")
    
    # Check if we have API keys
    if not os.getenv("ANTHROPIC_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        console.print("   [yellow]⚠️  No API keys - showing simulated conversation[/yellow]")
        
        # Simulated conversation
        demo_conversation = [
            {
                "user": "What's my current balance?",
                "agent": "Your current balance is $10,000.00. Is there anything else I can help you with?"
            },
            {
                "user": "I think I was charged twice for my monthly payment",
                "agent": "I've checked your recent transactions and found a potential duplicate charge. "
                         "On January 15th, there were two charges of $1,500.00 for 'Monthly payment - Solar financing' "
                         "within 5 minutes of each other. Would you like me to process a refund for the duplicate charge?"
            },
            {
                "user": "Yes, please refund the duplicate",
                "agent": "I've processed the refund for $1,500.00. The refund ID is ref_20250117123456. "
                         "The amount will be credited to your original payment method within 3-5 business days. "
                         "Is there anything else I can help you with?"
            }
        ]
        
        for exchange in demo_conversation:
            console.print(f"\n   [bold]You:[/bold] {exchange['user']}")
            console.print(f"   [bold blue]Agent:[/bold blue] {exchange['agent']}")
        
        return
    
    # Real agent conversation
    from agent.orchestrator import PaymentAgent
    
    agent = PaymentAgent()
    
    demo_prompts = [
        "What's my current balance?",
        "Show me my recent transactions",
        "I think I was charged twice for my monthly payment",
    ]
    
    print_step("2", "Starting conversation...")
    
    for prompt in demo_prompts:
        console.print(f"\n   [bold]You:[/bold] {prompt}")
        
        try:
            response = agent.chat(prompt, customer_id="cust_demo")
            console.print(f"   [bold blue]Agent:[/bold blue] {response}")
        except Exception as e:
            console.print(f"   [red]Error: {e}[/red]")


def demo_api_usage():
    """Show how to use the API."""
    print_header("🌐 API Usage Examples")
    
    console.print("""
[bold]Start the API server:[/bold]
    cd payment-agent-project
    uvicorn api.main:app --reload

[bold]Chat endpoint:[/bold]
    curl -X POST http://localhost:8000/chat \\
        -H "Content-Type: application/json" \\
        -d '{"message": "What is my balance?", "customer_id": "cust_demo"}'

[bold]RAG search endpoint:[/bold]
    curl -X POST http://localhost:8000/rag/search \\
        -H "Content-Type: application/json" \\
        -d '{"query": "refund policy", "k": 3}'

[bold]Metrics endpoint:[/bold]
    curl http://localhost:8000/metrics

[bold]Health check:[/bold]
    curl http://localhost:8000/health
    """)


def main():
    """Run the full demo."""
    console.print(Panel.fit(
        "[bold blue]PaymentGPT - AI Payment Agent Demo[/bold blue]\n"
        "Demonstrating MCP + RAG (Pinecone) + GenAI Observability",
        border_style="blue"
    ))
    
    console.print(f"\n[dim]Demo started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/dim]")
    
    # Check environment
    check_environment()
    
    # Run demos
    try:
        demo_mcp_servers()
    except Exception as e:
        console.print(f"[red]MCP demo error: {e}[/red]")
    
    try:
        demo_rag_pipeline()
    except Exception as e:
        console.print(f"[red]RAG demo error: {e}[/red]")
    
    try:
        demo_observability()
    except Exception as e:
        console.print(f"[red]Observability demo error: {e}[/red]")
    
    try:
        demo_agent_conversation()
    except Exception as e:
        console.print(f"[red]Agent demo error: {e}[/red]")
    
    demo_api_usage()
    
    # Summary
    print_header("✅ Demo Complete")
    
    console.print("""
[bold]What was demonstrated:[/bold]
    • MCP Server tools for payments, customers, and compliance
    • RAG pipeline with Pinecone vector database
    • GenAI observability with telemetry and metrics
    • Agent orchestration with LangChain/LangGraph
    • FastAPI REST API integration

[bold]Next steps:[/bold]
    1. Set your API keys in .env file
    2. Run: pip install -r requirements.txt
    3. Run: python demo.py (this script)
    4. Run: uvicorn api.main:app --reload
    5. Open: http://localhost:8000/docs
    """)


if __name__ == "__main__":
    main()
