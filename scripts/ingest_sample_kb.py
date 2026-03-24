"""
Sample KB Ingestion Script
===========================
Loads sample knowledge base articles into Pinecone.
Run this ONCE after setting up your Pinecone index.

Usage:
    python scripts/ingest_sample_kb.py

Requires .env to be configured with PINECONE_API_KEY + OPENAI_API_KEY.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.retrieval.pinecone_store import ingest_articles
from app.models.schemas import KBArticle, KnowledgeDomain

SAMPLE_ARTICLES = [
    KBArticle(
        article_id = "KB001",
        title      = "How to Reset Customer Portal Password",
        domain     = KnowledgeDomain.TECHNICAL,
        body       = """To reset a customer portal password, follow these steps:
1. Go to the customer portal login page and click 'Forgot Password'.
2. Enter the registered email address and click 'Send Reset Link'.
3. The customer will receive an email within 5 minutes with a reset link.
4. The link expires in 24 hours. If expired, repeat the process.
5. If the customer does not receive the email, check their spam folder.
6. For enterprise accounts, the reset may require admin approval.
Common issues: Email not found means the account was registered with a different address.
403 Forbidden after reset usually means the session cookie is stale — clear browser cache.""",
    ),
    KBArticle(
        article_id = "KB002",
        title      = "Billing Dispute Resolution Process",
        domain     = KnowledgeDomain.BILLING,
        body       = """When a customer raises a billing dispute:
1. Verify the charge in the billing system using the invoice number.
2. Check if the charge matches the customer's active subscription plan.
3. For overcharges, issue a credit note within 2 business days.
4. For duplicate charges, process a full refund within 5 business days.
5. Send the customer a confirmation email with the resolution timeline.
For annual plan customers, prorate refunds based on unused months.
Escalate disputes above INR 50,000 to the Finance team for approval.
Always document the dispute in the CRM case with the resolution action taken.""",
    ),
    KBArticle(
        article_id = "KB003",
        title      = "API Integration Troubleshooting Guide",
        domain     = KnowledgeDomain.TECHNICAL,
        body       = """Common API integration issues and resolutions:
401 Unauthorized: API key is missing, expired, or incorrect.
  - Ask customer to regenerate the API key from the developer portal.
  - Verify the key is being sent in the Authorization header, not the body.
429 Too Many Requests: Customer has exceeded their API rate limit.
  - Check their current plan's rate limit (Basic: 100/min, Pro: 1000/min).
  - Recommend upgrading plan or implementing request queuing.
500 Internal Server Error: Backend issue on our side.
  - Check the status page at status.yourcompany.com.
  - If ongoing, escalate to the engineering on-call team immediately.
CORS errors: Only occur in browser-based integrations.
  - Customer must whitelist their domain in the developer portal.""",
    ),
    KBArticle(
        article_id = "KB004",
        title      = "Subscription Plan Upgrade Process",
        domain     = KnowledgeDomain.BILLING,
        body       = """To process a subscription upgrade:
1. Verify the customer's current plan and usage metrics.
2. Explain the benefits of the target plan clearly.
3. Upgrades take effect immediately; customer is charged the prorated difference.
4. Downgrades take effect at the end of the current billing cycle.
5. Enterprise upgrades require a formal quote and signed order form.
Annual plan upgrades: calculate prorated cost = (remaining days / 365) * plan difference.
If customer is on a legacy plan, check eligibility for migration to new pricing.
Always send an upgrade confirmation email with the new invoice.""",
    ),
    KBArticle(
        article_id = "KB005",
        title      = "Data Export and GDPR Compliance Requests",
        domain     = KnowledgeDomain.GENERAL,
        body       = """Handling GDPR and data privacy requests:
Right to Access (SAR): Customer can request all their data.
  - Log the request in the Privacy Requests tracker.
  - We have 30 days to respond.
  - Export all customer data from CRM, billing, and usage systems.
Right to Erasure: Customer requests data deletion.
  - Verify identity before processing.
  - Deletion is irreversible. Confirm with the customer in writing.
  - Some data must be retained for legal/financial compliance (7 years).
Data Portability: Export data in machine-readable format (CSV/JSON).
  - Available from customer account settings for most data types.
Escalate all GDPR requests to the Data Protection Officer (DPO) team.""",
    ),
]


if __name__ == "__main__":
    print("=" * 60)
    print(" Ingesting sample KB articles into Pinecone")
    print("=" * 60)

    stats = ingest_articles(SAMPLE_ARTICLES)

    print(f"\n  Articles processed : {stats['articles_processed']}")
    print(f"  Vectors upserted   : {stats['vectors_upserted']}")
    print(f"  Errors             : {stats['errors']}")
    print("\n  Done! Run your FastAPI server and test with:")
    print('  curl -X POST http://localhost:8000/query \\')
    print('    -H "X-API-Key: your-key" \\')
    print('    -H "Content-Type: application/json" \\')
    print('    -d \'{"case_id":"500xx0000001ABC","question":'
          '"How do I reset a customer password?","domain":"technical"}\'')
    print("=" * 60)
