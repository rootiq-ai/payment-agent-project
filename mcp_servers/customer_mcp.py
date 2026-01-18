"""
Customer MCP Server
===================
Model Context Protocol server for customer data operations.
Exposes tools for customer profiles, preferences, and account management.

This demonstrates:
- Customer data management
- Profile retrieval and updates
- Integration with payment workflows
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Optional
from dataclasses import dataclass, asdict
from enum import Enum

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent


# ============================================================================
# Data Models
# ============================================================================

class CustomerTier(str, Enum):
    STANDARD = "standard"
    PREMIUM = "premium"
    VIP = "vip"


class CustomerStatus(str, Enum):
    ACTIVE = "active"
    SUSPENDED = "suspended"
    CLOSED = "closed"


@dataclass
class CustomerProfile:
    customer_id: str
    email: str
    first_name: str
    last_name: str
    phone: Optional[str]
    tier: CustomerTier
    status: CustomerStatus
    created_at: str
    address: dict
    preferences: dict
    metadata: dict

    def to_dict(self) -> dict:
        data = asdict(self)
        data["tier"] = self.tier.value
        data["status"] = self.status.value
        return data


@dataclass
class SupportTicket:
    ticket_id: str
    customer_id: str
    subject: str
    description: str
    status: str
    priority: str
    created_at: str
    resolved_at: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


# ============================================================================
# Mock Database
# ============================================================================

class CustomerDatabase:
    """In-memory customer database for demonstration."""
    
    def __init__(self):
        # Customer profiles
        self.customers: dict[str, CustomerProfile] = {
            "cust_demo": CustomerProfile(
                customer_id="cust_demo",
                email="demo@example.com",
                first_name="Sarah",
                last_name="Johnson",
                phone="+1-555-123-4567",
                tier=CustomerTier.PREMIUM,
                status=CustomerStatus.ACTIVE,
                created_at="2023-06-15T10:30:00Z",
                address={
                    "street": "123 Solar Lane",
                    "city": "Austin",
                    "state": "TX",
                    "zip": "78701",
                    "country": "US"
                },
                preferences={
                    "communication": "email",
                    "payment_reminder_days": 3,
                    "autopay_enabled": True,
                    "paperless_billing": True
                },
                metadata={
                    "acquisition_source": "referral",
                    "solar_system_size_kw": 8.5,
                    "installation_date": "2023-07-01",
                    "financing_term_months": 240,
                    "monthly_payment": 150.00
                }
            ),
            "cust_001": CustomerProfile(
                customer_id="cust_001",
                email="john.doe@example.com",
                first_name="John",
                last_name="Doe",
                phone="+1-555-987-6543",
                tier=CustomerTier.STANDARD,
                status=CustomerStatus.ACTIVE,
                created_at="2024-01-10T14:00:00Z",
                address={
                    "street": "456 Green Street",
                    "city": "San Francisco",
                    "state": "CA",
                    "zip": "94102",
                    "country": "US"
                },
                preferences={
                    "communication": "sms",
                    "payment_reminder_days": 5,
                    "autopay_enabled": False,
                    "paperless_billing": True
                },
                metadata={
                    "acquisition_source": "web",
                    "solar_system_size_kw": 6.0,
                    "installation_date": "2024-02-15",
                    "financing_term_months": 180,
                    "monthly_payment": 120.00
                }
            ),
        }
        
        # Support tickets
        self.tickets: list[SupportTicket] = [
            SupportTicket(
                ticket_id="tkt_001",
                customer_id="cust_demo",
                subject="Question about my bill",
                description="I noticed two charges for the same amount on my last statement.",
                status="open",
                priority="high",
                created_at=(datetime.now() - timedelta(hours=1)).isoformat()
            ),
            SupportTicket(
                ticket_id="tkt_002",
                customer_id="cust_demo",
                subject="Autopay setup",
                description="I'd like to enable autopay for my monthly payments.",
                status="resolved",
                priority="medium",
                created_at=(datetime.now() - timedelta(days=30)).isoformat(),
                resolved_at=(datetime.now() - timedelta(days=29)).isoformat()
            ),
        ]
    
    def get_customer(self, customer_id: str) -> Optional[CustomerProfile]:
        return self.customers.get(customer_id)
    
    def update_preferences(
        self, 
        customer_id: str, 
        preferences: dict
    ) -> Optional[CustomerProfile]:
        customer = self.customers.get(customer_id)
        if customer:
            customer.preferences.update(preferences)
        return customer
    
    def get_tickets(
        self, 
        customer_id: str, 
        status: Optional[str] = None
    ) -> list[SupportTicket]:
        tickets = [t for t in self.tickets if t.customer_id == customer_id]
        if status:
            tickets = [t for t in tickets if t.status == status]
        return tickets
    
    def create_ticket(
        self,
        customer_id: str,
        subject: str,
        description: str,
        priority: str = "medium"
    ) -> SupportTicket:
        ticket = SupportTicket(
            ticket_id=f"tkt_{len(self.tickets) + 1:03d}",
            customer_id=customer_id,
            subject=subject,
            description=description,
            status="open",
            priority=priority,
            created_at=datetime.now().isoformat()
        )
        self.tickets.append(ticket)
        return ticket


# ============================================================================
# MCP Server Implementation
# ============================================================================

db = CustomerDatabase()
server = Server("customer-mcp")


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List all available customer tools."""
    return [
        Tool(
            name="get_customer_profile",
            description="Get complete customer profile including contact info, preferences, and account details",
            inputSchema={
                "type": "object",
                "properties": {
                    "customer_id": {
                        "type": "string",
                        "description": "The unique customer identifier"
                    }
                },
                "required": ["customer_id"]
            }
        ),
        Tool(
            name="get_customer_preferences",
            description="Get customer communication and billing preferences",
            inputSchema={
                "type": "object",
                "properties": {
                    "customer_id": {
                        "type": "string",
                        "description": "The unique customer identifier"
                    }
                },
                "required": ["customer_id"]
            }
        ),
        Tool(
            name="update_preferences",
            description="Update customer preferences (communication, autopay, etc.)",
            inputSchema={
                "type": "object",
                "properties": {
                    "customer_id": {
                        "type": "string",
                        "description": "The unique customer identifier"
                    },
                    "preferences": {
                        "type": "object",
                        "description": "Preferences to update",
                        "properties": {
                            "communication": {
                                "type": "string",
                                "enum": ["email", "sms", "phone"]
                            },
                            "autopay_enabled": {
                                "type": "boolean"
                            },
                            "payment_reminder_days": {
                                "type": "integer"
                            },
                            "paperless_billing": {
                                "type": "boolean"
                            }
                        }
                    }
                },
                "required": ["customer_id", "preferences"]
            }
        ),
        Tool(
            name="get_financing_details",
            description="Get customer's solar financing details including loan terms and payment schedule",
            inputSchema={
                "type": "object",
                "properties": {
                    "customer_id": {
                        "type": "string",
                        "description": "The unique customer identifier"
                    }
                },
                "required": ["customer_id"]
            }
        ),
        Tool(
            name="get_support_tickets",
            description="Get customer's support ticket history",
            inputSchema={
                "type": "object",
                "properties": {
                    "customer_id": {
                        "type": "string",
                        "description": "The unique customer identifier"
                    },
                    "status": {
                        "type": "string",
                        "description": "Filter by ticket status",
                        "enum": ["open", "in_progress", "resolved", "closed"]
                    }
                },
                "required": ["customer_id"]
            }
        ),
        Tool(
            name="create_support_ticket",
            description="Create a new support ticket for the customer",
            inputSchema={
                "type": "object",
                "properties": {
                    "customer_id": {
                        "type": "string",
                        "description": "The customer ID"
                    },
                    "subject": {
                        "type": "string",
                        "description": "Ticket subject/title"
                    },
                    "description": {
                        "type": "string",
                        "description": "Detailed description of the issue"
                    },
                    "priority": {
                        "type": "string",
                        "description": "Ticket priority level",
                        "enum": ["low", "medium", "high", "urgent"],
                        "default": "medium"
                    }
                },
                "required": ["customer_id", "subject", "description"]
            }
        ),
        Tool(
            name="verify_customer_identity",
            description="Verify customer identity using provided information (for security purposes)",
            inputSchema={
                "type": "object",
                "properties": {
                    "customer_id": {
                        "type": "string",
                        "description": "The customer ID to verify"
                    },
                    "email": {
                        "type": "string",
                        "description": "Email to verify against"
                    },
                    "last_four_phone": {
                        "type": "string",
                        "description": "Last 4 digits of phone number"
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
        if name == "get_customer_profile":
            customer_id = arguments["customer_id"]
            customer = db.get_customer(customer_id)
            
            if customer is None:
                result = {
                    "success": False,
                    "error": f"Customer {customer_id} not found"
                }
            else:
                result = {
                    "success": True,
                    "customer": customer.to_dict()
                }
        
        elif name == "get_customer_preferences":
            customer_id = arguments["customer_id"]
            customer = db.get_customer(customer_id)
            
            if customer is None:
                result = {
                    "success": False,
                    "error": f"Customer {customer_id} not found"
                }
            else:
                result = {
                    "success": True,
                    "customer_id": customer_id,
                    "preferences": customer.preferences
                }
        
        elif name == "update_preferences":
            customer_id = arguments["customer_id"]
            preferences = arguments["preferences"]
            
            customer = db.update_preferences(customer_id, preferences)
            if customer is None:
                result = {
                    "success": False,
                    "error": f"Customer {customer_id} not found"
                }
            else:
                result = {
                    "success": True,
                    "message": "Preferences updated successfully",
                    "updated_preferences": customer.preferences
                }
        
        elif name == "get_financing_details":
            customer_id = arguments["customer_id"]
            customer = db.get_customer(customer_id)
            
            if customer is None:
                result = {
                    "success": False,
                    "error": f"Customer {customer_id} not found"
                }
            else:
                metadata = customer.metadata
                result = {
                    "success": True,
                    "customer_id": customer_id,
                    "financing": {
                        "solar_system_size_kw": metadata.get("solar_system_size_kw"),
                        "installation_date": metadata.get("installation_date"),
                        "financing_term_months": metadata.get("financing_term_months"),
                        "monthly_payment": metadata.get("monthly_payment"),
                        "loan_status": "active",
                        "next_payment_date": (datetime.now() + timedelta(days=15)).strftime("%Y-%m-%d")
                    }
                }
        
        elif name == "get_support_tickets":
            customer_id = arguments["customer_id"]
            status = arguments.get("status")
            
            tickets = db.get_tickets(customer_id, status)
            result = {
                "success": True,
                "customer_id": customer_id,
                "count": len(tickets),
                "tickets": [t.to_dict() for t in tickets]
            }
        
        elif name == "create_support_ticket":
            ticket = db.create_ticket(
                customer_id=arguments["customer_id"],
                subject=arguments["subject"],
                description=arguments["description"],
                priority=arguments.get("priority", "medium")
            )
            result = {
                "success": True,
                "message": "Support ticket created",
                "ticket": ticket.to_dict()
            }
        
        elif name == "verify_customer_identity":
            customer_id = arguments["customer_id"]
            customer = db.get_customer(customer_id)
            
            if customer is None:
                result = {
                    "success": False,
                    "verified": False,
                    "error": f"Customer {customer_id} not found"
                }
            else:
                # Simple verification logic
                verified = True
                checks = []
                
                if "email" in arguments:
                    email_match = arguments["email"].lower() == customer.email.lower()
                    verified = verified and email_match
                    checks.append({"check": "email", "passed": email_match})
                
                if "last_four_phone" in arguments and customer.phone:
                    phone_match = customer.phone.endswith(arguments["last_four_phone"])
                    verified = verified and phone_match
                    checks.append({"check": "phone_last4", "passed": phone_match})
                
                result = {
                    "success": True,
                    "verified": verified,
                    "customer_id": customer_id,
                    "checks_performed": checks
                }
        
        else:
            result = {"error": f"Unknown tool: {name}"}
        
        return [TextContent(type="text", text=json.dumps(result, indent=2))]
    
    except Exception as e:
        error_result = {
            "success": False,
            "error": str(e),
            "tool": name
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
