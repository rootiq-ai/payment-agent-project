"""
Payment MCP Server
==================
Model Context Protocol server for payment operations.
Exposes tools for balance checks, payments, refunds, and transaction history.

This demonstrates:
- MCP server implementation
- Tool definition with proper schemas
- Async operation handling
- Error handling and validation
"""

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum

# MCP imports
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
from pydantic import BaseModel, Field


# ============================================================================
# Data Models
# ============================================================================

class TransactionStatus(str, Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"
    REFUNDED = "refunded"
    CANCELLED = "cancelled"


class TransactionType(str, Enum):
    PAYMENT = "payment"
    REFUND = "refund"
    TRANSFER = "transfer"
    FEE = "fee"


@dataclass
class Transaction:
    transaction_id: str
    customer_id: str
    amount: float
    currency: str
    status: TransactionStatus
    transaction_type: TransactionType
    description: str
    created_at: str
    payment_method_id: Optional[str] = None
    metadata: Optional[dict] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PaymentResult:
    success: bool
    transaction_id: Optional[str]
    status: str
    message: str
    amount: Optional[float] = None
    balance_after: Optional[float] = None


# ============================================================================
# Mock Database (Replace with real DB in production)
# ============================================================================

class PaymentDatabase:
    """
    In-memory mock database for demonstration.
    In production, replace with PostgreSQL/DynamoDB connections.
    """
    
    def __init__(self):
        # Customer balances
        self.balances: dict[str, float] = {
            "cust_001": 5000.00,
            "cust_002": 12500.50,
            "cust_003": 750.25,
            "cust_demo": 10000.00,  # Demo customer
        }
        
        # Transaction history
        self.transactions: list[Transaction] = [
            Transaction(
                transaction_id="txn_001",
                customer_id="cust_demo",
                amount=250.00,
                currency="USD",
                status=TransactionStatus.COMPLETED,
                transaction_type=TransactionType.PAYMENT,
                description="Solar panel installation - deposit",
                created_at=(datetime.now() - timedelta(days=30)).isoformat(),
                payment_method_id="pm_visa_4242"
            ),
            Transaction(
                transaction_id="txn_002",
                customer_id="cust_demo",
                amount=1500.00,
                currency="USD",
                status=TransactionStatus.COMPLETED,
                transaction_type=TransactionType.PAYMENT,
                description="Monthly payment - Solar financing",
                created_at=(datetime.now() - timedelta(days=15)).isoformat(),
                payment_method_id="pm_visa_4242"
            ),
            Transaction(
                transaction_id="txn_003",
                customer_id="cust_demo",
                amount=1500.00,
                currency="USD",
                status=TransactionStatus.COMPLETED,
                transaction_type=TransactionType.PAYMENT,
                description="Monthly payment - Solar financing",
                created_at=(datetime.now() - timedelta(days=1)).isoformat(),
                payment_method_id="pm_visa_4242"
            ),
            # Duplicate charge for demo scenario
            Transaction(
                transaction_id="txn_004",
                customer_id="cust_demo",
                amount=1500.00,
                currency="USD",
                status=TransactionStatus.COMPLETED,
                transaction_type=TransactionType.PAYMENT,
                description="Monthly payment - Solar financing",
                created_at=(datetime.now() - timedelta(hours=2)).isoformat(),
                payment_method_id="pm_visa_4242"
            ),
        ]
        
        # Payment methods
        self.payment_methods: dict[str, dict] = {
            "pm_visa_4242": {
                "id": "pm_visa_4242",
                "customer_id": "cust_demo",
                "type": "card",
                "brand": "visa",
                "last4": "4242",
                "exp_month": 12,
                "exp_year": 2027,
                "is_default": True
            },
            "pm_bank_1234": {
                "id": "pm_bank_1234",
                "customer_id": "cust_demo",
                "type": "bank_account",
                "bank_name": "Chase",
                "last4": "1234",
                "is_default": False
            }
        }
    
    def get_balance(self, customer_id: str) -> Optional[float]:
        return self.balances.get(customer_id)
    
    def get_transactions(
        self, 
        customer_id: str, 
        limit: int = 10,
        status: Optional[TransactionStatus] = None
    ) -> list[Transaction]:
        txns = [t for t in self.transactions if t.customer_id == customer_id]
        if status:
            txns = [t for t in txns if t.status == status]
        return sorted(txns, key=lambda x: x.created_at, reverse=True)[:limit]
    
    def get_transaction(self, transaction_id: str) -> Optional[Transaction]:
        for txn in self.transactions:
            if txn.transaction_id == transaction_id:
                return txn
        return None
    
    def process_payment(
        self,
        customer_id: str,
        amount: float,
        description: str,
        payment_method_id: Optional[str] = None
    ) -> PaymentResult:
        # Validate customer exists
        if customer_id not in self.balances:
            return PaymentResult(
                success=False,
                transaction_id=None,
                status="failed",
                message=f"Customer {customer_id} not found"
            )
        
        # Create transaction
        txn_id = f"txn_{uuid.uuid4().hex[:8]}"
        txn = Transaction(
            transaction_id=txn_id,
            customer_id=customer_id,
            amount=amount,
            currency="USD",
            status=TransactionStatus.COMPLETED,
            transaction_type=TransactionType.PAYMENT,
            description=description,
            created_at=datetime.now().isoformat(),
            payment_method_id=payment_method_id
        )
        
        self.transactions.append(txn)
        # Note: In real system, balance would be affected differently
        # This is simplified for demo
        
        return PaymentResult(
            success=True,
            transaction_id=txn_id,
            status="completed",
            message="Payment processed successfully",
            amount=amount,
            balance_after=self.balances[customer_id]
        )
    
    def process_refund(
        self,
        original_transaction_id: str,
        amount: Optional[float] = None,
        reason: str = "Customer request"
    ) -> PaymentResult:
        # Find original transaction
        original_txn = self.get_transaction(original_transaction_id)
        if not original_txn:
            return PaymentResult(
                success=False,
                transaction_id=None,
                status="failed",
                message=f"Transaction {original_transaction_id} not found"
            )
        
        # Determine refund amount
        refund_amount = amount or original_txn.amount
        if refund_amount > original_txn.amount:
            return PaymentResult(
                success=False,
                transaction_id=None,
                status="failed",
                message=f"Refund amount ${refund_amount} exceeds original ${original_txn.amount}"
            )
        
        # Create refund transaction
        refund_txn_id = f"ref_{uuid.uuid4().hex[:8]}"
        refund_txn = Transaction(
            transaction_id=refund_txn_id,
            customer_id=original_txn.customer_id,
            amount=refund_amount,
            currency="USD",
            status=TransactionStatus.COMPLETED,
            transaction_type=TransactionType.REFUND,
            description=f"Refund for {original_transaction_id}: {reason}",
            created_at=datetime.now().isoformat(),
            metadata={"original_transaction_id": original_transaction_id}
        )
        
        self.transactions.append(refund_txn)
        
        # Update original transaction status
        original_txn.status = TransactionStatus.REFUNDED
        
        # Update balance
        self.balances[original_txn.customer_id] = (
            self.balances.get(original_txn.customer_id, 0) + refund_amount
        )
        
        return PaymentResult(
            success=True,
            transaction_id=refund_txn_id,
            status="completed",
            message=f"Refund of ${refund_amount} processed successfully",
            amount=refund_amount,
            balance_after=self.balances[original_txn.customer_id]
        )


# ============================================================================
# MCP Server Implementation
# ============================================================================

# Initialize database
db = PaymentDatabase()

# Create MCP server
server = Server("payment-mcp")


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List all available payment tools."""
    return [
        Tool(
            name="get_balance",
            description="Get the current account balance for a customer",
            inputSchema={
                "type": "object",
                "properties": {
                    "customer_id": {
                        "type": "string",
                        "description": "The unique customer identifier (e.g., 'cust_demo')"
                    }
                },
                "required": ["customer_id"]
            }
        ),
        Tool(
            name="get_transactions",
            description="Get transaction history for a customer. Returns recent transactions including payments, refunds, and fees.",
            inputSchema={
                "type": "object",
                "properties": {
                    "customer_id": {
                        "type": "string",
                        "description": "The unique customer identifier"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of transactions to return (default: 10)",
                        "default": 10
                    },
                    "status": {
                        "type": "string",
                        "description": "Filter by transaction status: pending, completed, failed, refunded",
                        "enum": ["pending", "completed", "failed", "refunded", "cancelled"]
                    }
                },
                "required": ["customer_id"]
            }
        ),
        Tool(
            name="get_transaction_details",
            description="Get detailed information about a specific transaction",
            inputSchema={
                "type": "object",
                "properties": {
                    "transaction_id": {
                        "type": "string",
                        "description": "The unique transaction identifier (e.g., 'txn_001')"
                    }
                },
                "required": ["transaction_id"]
            }
        ),
        Tool(
            name="process_payment",
            description="Process a payment for a customer. Requires compliance approval for amounts over $1000.",
            inputSchema={
                "type": "object",
                "properties": {
                    "customer_id": {
                        "type": "string",
                        "description": "The customer making the payment"
                    },
                    "amount": {
                        "type": "number",
                        "description": "Payment amount in USD"
                    },
                    "description": {
                        "type": "string",
                        "description": "Description of what this payment is for"
                    },
                    "payment_method_id": {
                        "type": "string",
                        "description": "Payment method to use (optional, uses default if not specified)"
                    }
                },
                "required": ["customer_id", "amount", "description"]
            }
        ),
        Tool(
            name="process_refund",
            description="Process a refund for a previous transaction. Can do full or partial refunds.",
            inputSchema={
                "type": "object",
                "properties": {
                    "transaction_id": {
                        "type": "string",
                        "description": "The original transaction ID to refund"
                    },
                    "amount": {
                        "type": "number",
                        "description": "Refund amount (optional, defaults to full refund)"
                    },
                    "reason": {
                        "type": "string",
                        "description": "Reason for the refund"
                    }
                },
                "required": ["transaction_id", "reason"]
            }
        ),
        Tool(
            name="get_payment_methods",
            description="Get available payment methods for a customer",
            inputSchema={
                "type": "object",
                "properties": {
                    "customer_id": {
                        "type": "string",
                        "description": "The customer ID to look up payment methods for"
                    }
                },
                "required": ["customer_id"]
            }
        ),
        Tool(
            name="detect_duplicate_charges",
            description="Check if there are potential duplicate charges for a customer within a time window",
            inputSchema={
                "type": "object",
                "properties": {
                    "customer_id": {
                        "type": "string",
                        "description": "The customer ID to check"
                    },
                    "hours": {
                        "type": "integer",
                        "description": "Time window in hours to check for duplicates (default: 24)",
                        "default": 24
                    }
                },
                "required": ["customer_id"]
            }
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls."""
    
    try:
        if name == "get_balance":
            customer_id = arguments["customer_id"]
            balance = db.get_balance(customer_id)
            
            if balance is None:
                result = {
                    "success": False,
                    "error": f"Customer {customer_id} not found"
                }
            else:
                result = {
                    "success": True,
                    "customer_id": customer_id,
                    "balance": balance,
                    "currency": "USD",
                    "as_of": datetime.now().isoformat()
                }
        
        elif name == "get_transactions":
            customer_id = arguments["customer_id"]
            limit = arguments.get("limit", 10)
            status = arguments.get("status")
            
            if status:
                status = TransactionStatus(status)
            
            transactions = db.get_transactions(customer_id, limit, status)
            result = {
                "success": True,
                "customer_id": customer_id,
                "count": len(transactions),
                "transactions": [t.to_dict() for t in transactions]
            }
        
        elif name == "get_transaction_details":
            transaction_id = arguments["transaction_id"]
            txn = db.get_transaction(transaction_id)
            
            if txn is None:
                result = {
                    "success": False,
                    "error": f"Transaction {transaction_id} not found"
                }
            else:
                result = {
                    "success": True,
                    "transaction": txn.to_dict()
                }
        
        elif name == "process_payment":
            payment_result = db.process_payment(
                customer_id=arguments["customer_id"],
                amount=arguments["amount"],
                description=arguments["description"],
                payment_method_id=arguments.get("payment_method_id")
            )
            result = asdict(payment_result)
        
        elif name == "process_refund":
            refund_result = db.process_refund(
                original_transaction_id=arguments["transaction_id"],
                amount=arguments.get("amount"),
                reason=arguments.get("reason", "Customer request")
            )
            result = asdict(refund_result)
        
        elif name == "get_payment_methods":
            customer_id = arguments["customer_id"]
            methods = [
                pm for pm in db.payment_methods.values()
                if pm["customer_id"] == customer_id
            ]
            result = {
                "success": True,
                "customer_id": customer_id,
                "payment_methods": methods
            }
        
        elif name == "detect_duplicate_charges":
            customer_id = arguments["customer_id"]
            hours = arguments.get("hours", 24)
            
            # Get recent transactions
            transactions = db.get_transactions(customer_id, limit=50)
            cutoff = datetime.now() - timedelta(hours=hours)
            
            # Filter to time window
            recent = [
                t for t in transactions
                if datetime.fromisoformat(t.created_at) > cutoff
                and t.transaction_type == TransactionType.PAYMENT
                and t.status == TransactionStatus.COMPLETED
            ]
            
            # Find duplicates (same amount within time window)
            duplicates = []
            seen_amounts = {}
            for txn in recent:
                key = f"{txn.amount}_{txn.description}"
                if key in seen_amounts:
                    duplicates.append({
                        "transaction_1": seen_amounts[key].to_dict(),
                        "transaction_2": txn.to_dict(),
                        "amount": txn.amount,
                        "potential_duplicate": True
                    })
                else:
                    seen_amounts[key] = txn
            
            result = {
                "success": True,
                "customer_id": customer_id,
                "time_window_hours": hours,
                "duplicates_found": len(duplicates),
                "duplicates": duplicates,
                "recommendation": "Review and refund if confirmed duplicate" if duplicates else "No duplicates detected"
            }
        
        else:
            result = {"error": f"Unknown tool: {name}"}
        
        return [TextContent(type="text", text=json.dumps(result, indent=2))]
    
    except Exception as e:
        error_result = {
            "success": False,
            "error": str(e),
            "tool": name,
            "arguments": arguments
        }
        return [TextContent(type="text", text=json.dumps(error_result, indent=2))]


# ============================================================================
# Server Entry Point
# ============================================================================

async def main():
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
