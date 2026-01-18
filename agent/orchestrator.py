"""
Agent Orchestrator
==================
Main agent orchestration logic using LangChain and LangGraph.
Coordinates between MCP servers, RAG, and LLM to handle complex payment workflows.

This demonstrates:
- Multi-agent orchestration
- Tool selection and routing
- Conversation memory management
- Complex workflow execution
"""

import os
import json
from typing import Annotated, TypedDict, Literal, Any, Optional
from datetime import datetime

from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool, StructuredTool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from pydantic import BaseModel, Field


# ============================================================================
# Configuration
# ============================================================================

SYSTEM_PROMPT = """You are PaymentGPT, an intelligent AI assistant for GoodLeap's payment platform. 
You help customers with payment-related inquiries, transactions, and account management for solar financing.

Your capabilities include:
1. **Balance & Transactions**: Check account balances, view transaction history, detect duplicate charges
2. **Payments & Refunds**: Process payments, initiate refunds, explain payment status
3. **Customer Support**: Access customer profiles, manage preferences, create support tickets
4. **Compliance**: Verify transaction compliance, check fraud indicators, explain policies

Guidelines:
- Always verify customer identity before making changes or sharing sensitive information
- For refunds or high-value operations, explain what you're checking (compliance, fraud) and why
- Be empathetic and helpful, especially when customers report issues like duplicate charges
- Proactively offer relevant information (e.g., mention autopay options when discussing payments)
- If you detect potential fraud or compliance issues, explain the situation clearly

When handling duplicate charge complaints:
1. First, use detect_duplicate_charges to check for duplicates
2. If found, verify with the customer which transaction should be refunded
3. Process the refund with proper documentation

Current customer context will be provided. Use the available tools to assist the customer.
"""


# ============================================================================
# State Definition
# ============================================================================

class AgentState(TypedDict):
    """State for the payment agent."""
    messages: list
    customer_id: Optional[str]
    current_intent: Optional[str]
    pending_actions: list
    metadata: dict


# ============================================================================
# Tool Definitions (MCP Tool Wrappers)
# ============================================================================

class GetBalanceInput(BaseModel):
    customer_id: str = Field(description="The unique customer identifier")

class GetTransactionsInput(BaseModel):
    customer_id: str = Field(description="The unique customer identifier")
    limit: int = Field(default=10, description="Maximum number of transactions")
    status: Optional[str] = Field(default=None, description="Filter by status")

class ProcessRefundInput(BaseModel):
    transaction_id: str = Field(description="Original transaction ID")
    amount: Optional[float] = Field(default=None, description="Refund amount (optional)")
    reason: str = Field(description="Reason for refund")

class DetectDuplicatesInput(BaseModel):
    customer_id: str = Field(description="Customer ID to check")
    hours: int = Field(default=24, description="Time window in hours")

class GetCustomerProfileInput(BaseModel):
    customer_id: str = Field(description="Customer ID")

class CheckFraudInput(BaseModel):
    transaction_id: str = Field(description="Transaction ID to check")
    customer_id: str = Field(description="Customer ID")
    amount: float = Field(description="Transaction amount")

class ProcessPaymentInput(BaseModel):
    customer_id: str = Field(description="Customer making the payment")
    amount: float = Field(description="Payment amount")
    description: str = Field(description="Payment description")

class CreateSupportTicketInput(BaseModel):
    customer_id: str = Field(description="Customer ID")
    subject: str = Field(description="Ticket subject")
    description: str = Field(description="Issue description")
    priority: str = Field(default="medium", description="Ticket priority")


# Simulated MCP tool implementations
# In production, these would call actual MCP servers via the MCP client

@tool("get_balance", args_schema=GetBalanceInput)
def get_balance(customer_id: str) -> dict:
    """Get the current account balance for a customer."""
    # Simulated response
    balances = {
        "cust_demo": 10000.00,
        "cust_001": 5000.00
    }
    
    if customer_id in balances:
        return {
            "success": True,
            "customer_id": customer_id,
            "balance": balances[customer_id],
            "currency": "USD",
            "as_of": datetime.now().isoformat()
        }
    return {"success": False, "error": f"Customer {customer_id} not found"}


@tool("get_transactions", args_schema=GetTransactionsInput)
def get_transactions(customer_id: str, limit: int = 10, status: Optional[str] = None) -> dict:
    """Get transaction history for a customer."""
    # Simulated response
    transactions = [
        {
            "transaction_id": "txn_001",
            "amount": 250.00,
            "status": "completed",
            "type": "payment",
            "description": "Solar panel installation - deposit",
            "date": "2024-12-15"
        },
        {
            "transaction_id": "txn_002",
            "amount": 1500.00,
            "status": "completed",
            "type": "payment",
            "description": "Monthly payment - Solar financing",
            "date": "2025-01-01"
        },
        {
            "transaction_id": "txn_003",
            "amount": 1500.00,
            "status": "completed",
            "type": "payment",
            "description": "Monthly payment - Solar financing",
            "date": "2025-01-15"
        },
        {
            "transaction_id": "txn_004",
            "amount": 1500.00,
            "status": "completed",
            "type": "payment",
            "description": "Monthly payment - Solar financing",
            "date": "2025-01-15"
        }
    ]
    
    return {
        "success": True,
        "customer_id": customer_id,
        "count": len(transactions[:limit]),
        "transactions": transactions[:limit]
    }


@tool("detect_duplicate_charges", args_schema=DetectDuplicatesInput)
def detect_duplicate_charges(customer_id: str, hours: int = 24) -> dict:
    """Detect potential duplicate charges for a customer."""
    # Simulated duplicate detection
    return {
        "success": True,
        "customer_id": customer_id,
        "time_window_hours": hours,
        "duplicates_found": 1,
        "duplicates": [
            {
                "transaction_1": {
                    "transaction_id": "txn_003",
                    "amount": 1500.00,
                    "date": "2025-01-15T10:00:00"
                },
                "transaction_2": {
                    "transaction_id": "txn_004",
                    "amount": 1500.00,
                    "date": "2025-01-15T10:05:00"
                },
                "amount": 1500.00,
                "potential_duplicate": True
            }
        ],
        "recommendation": "Review and refund if confirmed duplicate"
    }


@tool("process_refund", args_schema=ProcessRefundInput)
def process_refund(transaction_id: str, reason: str, amount: Optional[float] = None) -> dict:
    """Process a refund for a transaction."""
    return {
        "success": True,
        "refund_id": f"ref_{datetime.now().strftime('%Y%m%d%H%M%S')}",
        "original_transaction_id": transaction_id,
        "amount": amount or 1500.00,
        "status": "completed",
        "reason": reason,
        "message": "Refund processed successfully. Amount will be credited within 3-5 business days."
    }


@tool("get_customer_profile", args_schema=GetCustomerProfileInput)
def get_customer_profile(customer_id: str) -> dict:
    """Get customer profile information."""
    return {
        "success": True,
        "customer": {
            "customer_id": customer_id,
            "name": "Sarah Johnson",
            "email": "demo@example.com",
            "phone": "+1-555-123-4567",
            "tier": "premium",
            "status": "active",
            "solar_system": "8.5 kW",
            "monthly_payment": 150.00,
            "autopay_enabled": True
        }
    }


@tool("check_fraud", args_schema=CheckFraudInput)
def check_fraud(transaction_id: str, customer_id: str, amount: float) -> dict:
    """Check a transaction for fraud indicators."""
    return {
        "success": True,
        "fraud_check": {
            "transaction_id": transaction_id,
            "is_suspicious": False,
            "fraud_score": 15,
            "flags": [],
            "recommendation": "Transaction appears legitimate - approve",
            "requires_manual_review": False
        }
    }


@tool("process_payment", args_schema=ProcessPaymentInput)
def process_payment(customer_id: str, amount: float, description: str) -> dict:
    """Process a payment for a customer."""
    return {
        "success": True,
        "transaction_id": f"txn_{datetime.now().strftime('%Y%m%d%H%M%S')}",
        "amount": amount,
        "status": "completed",
        "message": f"Payment of ${amount} processed successfully"
    }


@tool("create_support_ticket", args_schema=CreateSupportTicketInput)
def create_support_ticket(
    customer_id: str, 
    subject: str, 
    description: str, 
    priority: str = "medium"
) -> dict:
    """Create a support ticket for the customer."""
    return {
        "success": True,
        "ticket_id": f"tkt_{datetime.now().strftime('%Y%m%d%H%M%S')}",
        "status": "open",
        "priority": priority,
        "message": f"Support ticket created. A representative will contact you within 24 hours."
    }


@tool("search_payment_knowledge")
def search_payment_knowledge(query: str) -> dict:
    """Search the payment knowledge base for policies and procedures."""
    # Simulated RAG response
    knowledge_base = {
        "refund": {
            "policy": "Refunds are processed within 3-5 business days. Full refunds available for duplicate charges. Partial refunds available for billing disputes.",
            "process": "1. Verify duplicate charge 2. Check compliance 3. Process refund 4. Send confirmation"
        },
        "autopay": {
            "policy": "Autopay can be enabled/disabled at any time. Payments are processed on the same day each month.",
            "benefits": "5% discount on monthly payments when autopay is enabled."
        },
        "solar financing": {
            "terms": "Financing terms range from 5-25 years with competitive APR rates.",
            "early_payoff": "No prepayment penalties. Pay off your loan early with no additional fees."
        }
    }
    
    # Simple keyword matching (RAG would use vector search)
    results = []
    query_lower = query.lower()
    for topic, info in knowledge_base.items():
        if topic in query_lower or any(word in query_lower for word in topic.split()):
            results.append({"topic": topic, **info})
    
    return {
        "success": True,
        "query": query,
        "results": results if results else [{"note": "No specific policy found. Please contact support."}]
    }


# ============================================================================
# Tool Collection
# ============================================================================

TOOLS = [
    get_balance,
    get_transactions,
    detect_duplicate_charges,
    process_refund,
    get_customer_profile,
    check_fraud,
    process_payment,
    create_support_ticket,
    search_payment_knowledge
]


# ============================================================================
# Agent Graph
# ============================================================================

def create_payment_agent(model_name: str = "claude-sonnet-4-20250514"):
    """Create the payment agent graph."""
    
    # Initialize LLM
    if "claude" in model_name.lower():
        llm = ChatAnthropic(
            model=model_name,
            temperature=0.1,
            max_tokens=4096
        )
    else:
        llm = ChatOpenAI(
            model=model_name,
            temperature=0.1,
            max_tokens=4096
        )
    
    # Bind tools to LLM
    llm_with_tools = llm.bind_tools(TOOLS)
    
    # Define nodes
    def should_continue(state: AgentState) -> Literal["tools", "end"]:
        """Determine if we should continue to tools or end."""
        messages = state["messages"]
        last_message = messages[-1]
        
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        return "end"
    
    def call_model(state: AgentState) -> dict:
        """Call the LLM with current state."""
        messages = state["messages"]
        
        # Build system message with context
        system_content = SYSTEM_PROMPT
        if state.get("customer_id"):
            system_content += f"\n\nCurrent customer: {state['customer_id']}"
        
        # Prepend system message
        full_messages = [SystemMessage(content=system_content)] + messages
        
        # Call LLM
        response = llm_with_tools.invoke(full_messages)
        
        return {"messages": [response]}
    
    # Create tool node
    tool_node = ToolNode(TOOLS)
    
    # Build graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_node)
    
    # Set entry point
    workflow.set_entry_point("agent")
    
    # Add conditional edges
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "end": END
        }
    )
    
    # Tools always go back to agent
    workflow.add_edge("tools", "agent")
    
    # Compile with memory
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    
    return app


# ============================================================================
# Agent Interface
# ============================================================================

class PaymentAgent:
    """High-level interface for the payment agent."""
    
    def __init__(self, model_name: str = "claude-sonnet-4-20250514"):
        self.app = create_payment_agent(model_name)
        self.thread_id = "default"
    
    def chat(
        self, 
        message: str, 
        customer_id: Optional[str] = None,
        thread_id: Optional[str] = None
    ) -> str:
        """Send a message and get a response."""
        
        # Use provided thread_id or default
        config = {"configurable": {"thread_id": thread_id or self.thread_id}}
        
        # Build initial state
        state = {
            "messages": [HumanMessage(content=message)],
            "customer_id": customer_id or "cust_demo",
            "current_intent": None,
            "pending_actions": [],
            "metadata": {}
        }
        
        # Run agent
        result = self.app.invoke(state, config)
        
        # Extract final response
        final_message = result["messages"][-1]
        
        if hasattr(final_message, "content"):
            return final_message.content
        return str(final_message)
    
    def reset_conversation(self, thread_id: Optional[str] = None):
        """Reset conversation history."""
        self.thread_id = thread_id or f"thread_{datetime.now().strftime('%Y%m%d%H%M%S')}"


# ============================================================================
# Demo / CLI Interface
# ============================================================================

def run_demo():
    """Run interactive demo of the payment agent."""
    print("=" * 60)
    print("PaymentGPT - AI Payment Assistant Demo")
    print("=" * 60)
    print("\nInitializing agent...")
    
    # Check for API key
    if not os.getenv("ANTHROPIC_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        print("\n⚠️  No API key found. Set ANTHROPIC_API_KEY or OPENAI_API_KEY")
        print("Running in simulation mode...\n")
    
    agent = PaymentAgent()
    
    print("\nAgent ready! You are logged in as customer: cust_demo (Sarah Johnson)")
    print("Type 'quit' to exit, 'reset' to start a new conversation\n")
    print("-" * 60)
    
    # Demo scenarios
    demo_prompts = [
        "What's my current balance?",
        "Show me my recent transactions",
        "I think I was charged twice for my monthly payment!",
        "Yes, please refund the duplicate charge",
        "What's your refund policy?",
    ]
    
    print("\nSuggested demo prompts:")
    for i, prompt in enumerate(demo_prompts, 1):
        print(f"  {i}. {prompt}")
    print("-" * 60 + "\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() == "quit":
                print("\nGoodbye!")
                break
            
            if user_input.lower() == "reset":
                agent.reset_conversation()
                print("\n[Conversation reset]\n")
                continue
            
            # Get response
            print("\nPaymentGPT: ", end="")
            response = agent.chat(user_input, customer_id="cust_demo")
            print(response)
            print()
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\n[Error: {e}]\n")


if __name__ == "__main__":
    run_demo()
