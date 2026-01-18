"""
Compliance MCP Server
=====================
Model Context Protocol server for risk assessment and compliance operations.
Exposes tools for fraud detection, compliance checks, and risk scoring.

This demonstrates:
- Financial compliance workflows
- Fraud detection patterns
- Risk scoring algorithms
- Regulatory compliance checks
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Optional
from dataclasses import dataclass, asdict
from enum import Enum
import random

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent


# ============================================================================
# Data Models
# ============================================================================

class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ComplianceStatus(str, Enum):
    COMPLIANT = "compliant"
    REVIEW_REQUIRED = "review_required"
    NON_COMPLIANT = "non_compliant"


@dataclass
class RiskAssessment:
    customer_id: str
    risk_level: RiskLevel
    risk_score: float  # 0-100
    factors: list[dict]
    recommendation: str
    assessed_at: str

    def to_dict(self) -> dict:
        data = asdict(self)
        data["risk_level"] = self.risk_level.value
        return data


@dataclass
class FraudCheck:
    transaction_id: str
    is_suspicious: bool
    fraud_score: float  # 0-100
    flags: list[str]
    recommendation: str
    requires_manual_review: bool

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ComplianceCheck:
    check_type: str
    status: ComplianceStatus
    details: dict
    checked_at: str

    def to_dict(self) -> dict:
        data = asdict(self)
        data["status"] = self.status.value
        return data


# ============================================================================
# Compliance Engine
# ============================================================================

class ComplianceEngine:
    """
    Compliance and risk assessment engine.
    In production, this would integrate with:
    - Fraud detection ML models
    - KYC/AML services
    - Credit bureaus
    - Regulatory databases
    """
    
    # Known patterns for demo
    SUSPICIOUS_PATTERNS = {
        "rapid_transactions": "Multiple transactions in short time window",
        "amount_spike": "Transaction amount significantly higher than usual",
        "new_payment_method": "Transaction using newly added payment method",
        "geo_anomaly": "Transaction from unusual location",
        "duplicate_charge": "Potential duplicate charge detected"
    }
    
    # Customer risk profiles (for demo)
    CUSTOMER_RISK_PROFILES = {
        "cust_demo": {
            "base_risk": 15,
            "account_age_months": 18,
            "payment_history": "excellent",
            "verification_level": "full"
        },
        "cust_001": {
            "base_risk": 25,
            "account_age_months": 6,
            "payment_history": "good",
            "verification_level": "basic"
        }
    }
    
    def assess_customer_risk(self, customer_id: str) -> RiskAssessment:
        """Assess overall risk level for a customer."""
        
        profile = self.CUSTOMER_RISK_PROFILES.get(customer_id, {
            "base_risk": 50,
            "account_age_months": 0,
            "payment_history": "unknown",
            "verification_level": "none"
        })
        
        factors = []
        risk_score = profile["base_risk"]
        
        # Account age factor
        if profile["account_age_months"] < 3:
            risk_score += 20
            factors.append({
                "factor": "new_account",
                "impact": +20,
                "detail": f"Account is only {profile['account_age_months']} months old"
            })
        elif profile["account_age_months"] > 12:
            risk_score -= 10
            factors.append({
                "factor": "established_account",
                "impact": -10,
                "detail": f"Account is {profile['account_age_months']} months old"
            })
        
        # Payment history factor
        if profile["payment_history"] == "excellent":
            risk_score -= 15
            factors.append({
                "factor": "payment_history",
                "impact": -15,
                "detail": "Excellent payment history"
            })
        elif profile["payment_history"] == "poor":
            risk_score += 25
            factors.append({
                "factor": "payment_history",
                "impact": +25,
                "detail": "History of late/missed payments"
            })
        
        # Verification level
        if profile["verification_level"] == "full":
            risk_score -= 10
            factors.append({
                "factor": "verification",
                "impact": -10,
                "detail": "Full identity verification completed"
            })
        elif profile["verification_level"] == "none":
            risk_score += 30
            factors.append({
                "factor": "verification",
                "impact": +30,
                "detail": "No identity verification on file"
            })
        
        # Clamp score
        risk_score = max(0, min(100, risk_score))
        
        # Determine risk level
        if risk_score < 25:
            risk_level = RiskLevel.LOW
            recommendation = "Standard processing approved"
        elif risk_score < 50:
            risk_level = RiskLevel.MEDIUM
            recommendation = "Standard processing with monitoring"
        elif risk_score < 75:
            risk_level = RiskLevel.HIGH
            recommendation = "Enhanced verification recommended"
        else:
            risk_level = RiskLevel.CRITICAL
            recommendation = "Manual review required before processing"
        
        return RiskAssessment(
            customer_id=customer_id,
            risk_level=risk_level,
            risk_score=risk_score,
            factors=factors,
            recommendation=recommendation,
            assessed_at=datetime.now().isoformat()
        )
    
    def check_fraud(
        self, 
        transaction_id: str,
        customer_id: str,
        amount: float,
        payment_method_id: Optional[str] = None
    ) -> FraudCheck:
        """Check a transaction for potential fraud."""
        
        flags = []
        fraud_score = 0
        
        # Amount-based checks
        if amount > 5000:
            fraud_score += 25
            flags.append("high_value_transaction")
        elif amount > 10000:
            fraud_score += 40
            flags.append("very_high_value_transaction")
        
        # Simulate pattern detection
        # In production, this would check actual transaction history
        if random.random() < 0.1:  # 10% chance for demo
            fraud_score += 15
            flags.append("velocity_check_triggered")
        
        # New payment method check
        if payment_method_id and "new" in payment_method_id.lower():
            fraud_score += 20
            flags.append("new_payment_method")
        
        # Determine result
        is_suspicious = fraud_score >= 40
        requires_manual_review = fraud_score >= 60
        
        if fraud_score < 25:
            recommendation = "Transaction appears legitimate - approve"
        elif fraud_score < 50:
            recommendation = "Minor flags detected - approve with monitoring"
        elif fraud_score < 75:
            recommendation = "Significant flags - manual review recommended"
        else:
            recommendation = "High fraud risk - block and investigate"
        
        return FraudCheck(
            transaction_id=transaction_id,
            is_suspicious=is_suspicious,
            fraud_score=min(100, fraud_score),
            flags=flags,
            recommendation=recommendation,
            requires_manual_review=requires_manual_review
        )
    
    def check_pci_compliance(self, customer_id: str) -> ComplianceCheck:
        """Check PCI-DSS compliance status."""
        
        # Simulated PCI compliance check
        # In production, this would verify actual compliance controls
        
        return ComplianceCheck(
            check_type="PCI-DSS",
            status=ComplianceStatus.COMPLIANT,
            details={
                "version": "4.0",
                "last_audit": "2024-06-15",
                "next_audit_due": "2025-06-15",
                "controls_verified": [
                    "encryption_at_rest",
                    "encryption_in_transit",
                    "access_controls",
                    "audit_logging"
                ]
            },
            checked_at=datetime.now().isoformat()
        )
    
    def check_aml(self, customer_id: str, amount: float) -> ComplianceCheck:
        """Check Anti-Money Laundering compliance."""
        
        # AML thresholds
        if amount >= 10000:
            status = ComplianceStatus.REVIEW_REQUIRED
            details = {
                "reason": "Transaction meets CTR threshold",
                "threshold": 10000,
                "action_required": "File Currency Transaction Report",
                "deadline": "15 days"
            }
        elif amount >= 3000:
            status = ComplianceStatus.COMPLIANT
            details = {
                "note": "Transaction below CTR threshold",
                "monitoring": "Standard SAR monitoring applies"
            }
        else:
            status = ComplianceStatus.COMPLIANT
            details = {
                "note": "Standard transaction - no special reporting required"
            }
        
        return ComplianceCheck(
            check_type="AML/BSA",
            status=status,
            details=details,
            checked_at=datetime.now().isoformat()
        )
    
    def check_transaction_limits(
        self, 
        customer_id: str, 
        amount: float,
        transaction_type: str = "payment"
    ) -> dict:
        """Check if transaction is within allowed limits."""
        
        # Demo limits
        limits = {
            "daily_limit": 25000,
            "single_transaction_limit": 10000,
            "monthly_limit": 100000
        }
        
        # Simulated current usage
        current_daily = 1500  # Would come from actual DB
        current_monthly = 5000
        
        checks = {
            "within_single_limit": amount <= limits["single_transaction_limit"],
            "within_daily_limit": (current_daily + amount) <= limits["daily_limit"],
            "within_monthly_limit": (current_monthly + amount) <= limits["monthly_limit"]
        }
        
        all_passed = all(checks.values())
        
        return {
            "approved": all_passed,
            "amount": amount,
            "limits": limits,
            "current_usage": {
                "daily": current_daily,
                "monthly": current_monthly
            },
            "checks": checks,
            "message": "Transaction approved" if all_passed else "Transaction exceeds limits"
        }


# ============================================================================
# MCP Server Implementation
# ============================================================================

engine = ComplianceEngine()
server = Server("compliance-mcp")


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List all available compliance tools."""
    return [
        Tool(
            name="assess_customer_risk",
            description="Get comprehensive risk assessment for a customer including risk score and factors",
            inputSchema={
                "type": "object",
                "properties": {
                    "customer_id": {
                        "type": "string",
                        "description": "The customer ID to assess"
                    }
                },
                "required": ["customer_id"]
            }
        ),
        Tool(
            name="check_fraud",
            description="Check a transaction for potential fraud indicators",
            inputSchema={
                "type": "object",
                "properties": {
                    "transaction_id": {
                        "type": "string",
                        "description": "The transaction ID to check"
                    },
                    "customer_id": {
                        "type": "string",
                        "description": "The customer making the transaction"
                    },
                    "amount": {
                        "type": "number",
                        "description": "Transaction amount in USD"
                    },
                    "payment_method_id": {
                        "type": "string",
                        "description": "Payment method being used"
                    }
                },
                "required": ["transaction_id", "customer_id", "amount"]
            }
        ),
        Tool(
            name="check_pci_compliance",
            description="Verify PCI-DSS compliance status for payment processing",
            inputSchema={
                "type": "object",
                "properties": {
                    "customer_id": {
                        "type": "string",
                        "description": "Customer ID for context"
                    }
                },
                "required": ["customer_id"]
            }
        ),
        Tool(
            name="check_aml_compliance",
            description="Check Anti-Money Laundering compliance for a transaction",
            inputSchema={
                "type": "object",
                "properties": {
                    "customer_id": {
                        "type": "string",
                        "description": "The customer ID"
                    },
                    "amount": {
                        "type": "number",
                        "description": "Transaction amount in USD"
                    }
                },
                "required": ["customer_id", "amount"]
            }
        ),
        Tool(
            name="check_transaction_limits",
            description="Verify transaction is within customer's allowed limits",
            inputSchema={
                "type": "object",
                "properties": {
                    "customer_id": {
                        "type": "string",
                        "description": "The customer ID"
                    },
                    "amount": {
                        "type": "number",
                        "description": "Transaction amount in USD"
                    },
                    "transaction_type": {
                        "type": "string",
                        "description": "Type of transaction",
                        "enum": ["payment", "refund", "transfer"]
                    }
                },
                "required": ["customer_id", "amount"]
            }
        ),
        Tool(
            name="approve_high_value_transaction",
            description="Request approval for high-value transactions that exceed standard limits",
            inputSchema={
                "type": "object",
                "properties": {
                    "customer_id": {
                        "type": "string",
                        "description": "The customer ID"
                    },
                    "amount": {
                        "type": "number",
                        "description": "Transaction amount"
                    },
                    "reason": {
                        "type": "string",
                        "description": "Reason for the high-value transaction"
                    },
                    "supporting_documents": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of supporting document IDs"
                    }
                },
                "required": ["customer_id", "amount", "reason"]
            }
        ),
        Tool(
            name="flag_suspicious_activity",
            description="Flag and report suspicious activity for investigation",
            inputSchema={
                "type": "object",
                "properties": {
                    "customer_id": {
                        "type": "string",
                        "description": "The customer ID"
                    },
                    "activity_type": {
                        "type": "string",
                        "description": "Type of suspicious activity",
                        "enum": ["fraud", "aml", "identity", "other"]
                    },
                    "description": {
                        "type": "string",
                        "description": "Description of the suspicious activity"
                    },
                    "transaction_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Related transaction IDs"
                    }
                },
                "required": ["customer_id", "activity_type", "description"]
            }
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls."""
    
    try:
        if name == "assess_customer_risk":
            assessment = engine.assess_customer_risk(arguments["customer_id"])
            result = {
                "success": True,
                "assessment": assessment.to_dict()
            }
        
        elif name == "check_fraud":
            fraud_check = engine.check_fraud(
                transaction_id=arguments["transaction_id"],
                customer_id=arguments["customer_id"],
                amount=arguments["amount"],
                payment_method_id=arguments.get("payment_method_id")
            )
            result = {
                "success": True,
                "fraud_check": fraud_check.to_dict()
            }
        
        elif name == "check_pci_compliance":
            compliance = engine.check_pci_compliance(arguments["customer_id"])
            result = {
                "success": True,
                "compliance": compliance.to_dict()
            }
        
        elif name == "check_aml_compliance":
            compliance = engine.check_aml(
                customer_id=arguments["customer_id"],
                amount=arguments["amount"]
            )
            result = {
                "success": True,
                "compliance": compliance.to_dict()
            }
        
        elif name == "check_transaction_limits":
            limits_check = engine.check_transaction_limits(
                customer_id=arguments["customer_id"],
                amount=arguments["amount"],
                transaction_type=arguments.get("transaction_type", "payment")
            )
            result = {
                "success": True,
                **limits_check
            }
        
        elif name == "approve_high_value_transaction":
            # Simulated approval workflow
            amount = arguments["amount"]
            risk = engine.assess_customer_risk(arguments["customer_id"])
            
            # Auto-approve for low-risk customers under $50k
            if risk.risk_level in [RiskLevel.LOW, RiskLevel.MEDIUM] and amount < 50000:
                approved = True
                message = "Auto-approved based on customer risk profile"
            else:
                approved = False
                message = "Requires manual review by compliance team"
            
            result = {
                "success": True,
                "approved": approved,
                "amount": amount,
                "risk_level": risk.risk_level.value,
                "message": message,
                "reference_id": f"apr_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            }
        
        elif name == "flag_suspicious_activity":
            # Create SAR (Suspicious Activity Report)
            sar_id = f"SAR_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            result = {
                "success": True,
                "sar_id": sar_id,
                "status": "filed",
                "customer_id": arguments["customer_id"],
                "activity_type": arguments["activity_type"],
                "description": arguments["description"],
                "transaction_ids": arguments.get("transaction_ids", []),
                "filed_at": datetime.now().isoformat(),
                "message": "Suspicious Activity Report filed successfully. Compliance team notified."
            }
        
        else:
            result = {"success": False, "error": f"Unknown tool: {name}"}
        
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
