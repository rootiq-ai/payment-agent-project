"""
RAG Pipeline
============
Retrieval-Augmented Generation pipeline for payment knowledge.
Indexes and retrieves payment policies, FAQs, and compliance documents.

This demonstrates:
- Document loading and chunking
- Vector embedding and indexing with Pinecone
- Semantic search with reranking
- RAG integration with LLM
- Hybrid search (dense + sparse)
"""

import os
from typing import Optional, List, Dict, Any
from pathlib import Path
from datetime import datetime
import hashlib
import time

from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore


# ============================================================================
# Configuration
# ============================================================================

PINECONE_INDEX_NAME = "payment-knowledge"
PINECONE_NAMESPACE = "goodleap-payments"

# Embedding model
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536

# Chunking settings
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Pinecone serverless spec
PINECONE_CLOUD = "aws"
PINECONE_REGION = "us-east-1"


# ============================================================================
# Sample Knowledge Base Documents
# ============================================================================

PAYMENT_KNOWLEDGE_DOCS = [
    # Refund Policies
    Document(
        page_content="""
        # Refund Policy
        
        GoodLeap processes refunds within 3-5 business days of approval. 
        
        ## Eligible Refunds
        - Duplicate charges: Full refund of the duplicate amount
        - Billing errors: Full refund of the error amount
        - Service cancellation: Prorated refund based on contract terms
        - Overpayments: Full refund of excess amount
        
        ## Refund Process
        1. Customer reports issue through support or AI assistant
        2. System detects and verifies duplicate/error
        3. Compliance check is performed automatically
        4. Refund is processed to original payment method
        5. Confirmation email sent to customer
        
        ## Timeframes
        - Credit card refunds: 3-5 business days
        - Bank transfer refunds: 5-7 business days
        - Check refunds (rare): 10-14 business days
        
        Note: Refunds over $5,000 require additional compliance review.
        """,
        metadata={
            "topic": "refund_policy",
            "category": "policies",
            "last_updated": "2024-12-01"
        }
    ),
    
    # Autopay Information
    Document(
        page_content="""
        # Autopay Program
        
        Enroll in autopay and enjoy hassle-free payments with added benefits.
        
        ## Benefits
        - 5% discount on monthly payment amount
        - Never miss a payment or incur late fees
        - Flexible payment date selection
        - Easy modification or cancellation anytime
        
        ## How to Enroll
        1. Log into your account or contact support
        2. Verify your preferred payment method
        3. Select your desired payment date (1st-28th of month)
        4. Confirm enrollment
        
        ## Managing Autopay
        - Change payment date: Update anytime, effective next billing cycle
        - Update payment method: Changes take effect within 24 hours
        - Cancel autopay: Request by 3 days before next payment date
        
        ## Autopay Failures
        If an autopay payment fails:
        1. We'll notify you immediately via email/SMS
        2. We'll retry once after 3 days
        3. If retry fails, autopay is paused until you update payment info
        """,
        metadata={
            "topic": "autopay",
            "category": "features",
            "last_updated": "2024-11-15"
        }
    ),
    
    # Solar Financing Terms
    Document(
        page_content="""
        # Solar Financing Terms
        
        GoodLeap offers flexible financing options for residential solar installations.
        
        ## Loan Terms
        - Term lengths: 5, 10, 15, 20, or 25 years
        - APR range: 3.99% - 8.99% based on credit profile
        - Loan amounts: $5,000 - $100,000
        - No prepayment penalties
        
        ## Payment Options
        - Monthly payments: Due on the same day each month
        - Bi-weekly payments: Available for faster payoff
        - One-time extra payments: Apply directly to principal
        
        ## Early Payoff
        - No prepayment penalties ever
        - Request payoff quote through your account or support
        - Full payoff statement provided within 24 hours
        - Lien release processed within 30 days of final payment
        
        ## Tax Benefits
        - Federal Solar Tax Credit (ITC): Up to 30% of system cost
        - State incentives vary by location
        - Consult tax professional for personalized advice
        """,
        metadata={
            "topic": "solar_financing",
            "category": "products",
            "last_updated": "2024-10-01"
        }
    ),
    
    # Dispute Resolution
    Document(
        page_content="""
        # Dispute Resolution Process
        
        If you have a billing dispute, we're here to help resolve it quickly.
        
        ## Types of Disputes
        - Incorrect charge amount
        - Unauthorized transaction
        - Service not received
        - Duplicate billing
        - Contract disagreement
        
        ## How to File a Dispute
        1. Contact support via chat, phone, or email
        2. Provide transaction details and reason for dispute
        3. Submit any supporting documentation
        4. Receive dispute case number
        
        ## Resolution Timeline
        - Initial review: Within 2 business days
        - Investigation: 5-10 business days
        - Resolution: Within 30 days per CFPB guidelines
        
        ## Your Rights
        - You may withhold payment of disputed amount during investigation
        - No late fees on disputed amounts
        - Written response required within 30 days
        - If error confirmed, correction within 2 business days
        
        ## Escalation
        If unsatisfied with resolution:
        1. Request supervisor review
        2. Contact compliance team
        3. File complaint with CFPB (consumerfinance.gov)
        """,
        metadata={
            "topic": "dispute_resolution",
            "category": "support",
            "last_updated": "2024-09-15"
        }
    ),
    
    # Payment Methods
    Document(
        page_content="""
        # Accepted Payment Methods
        
        GoodLeap accepts multiple payment methods for your convenience.
        
        ## Credit/Debit Cards
        - Visa, Mastercard, American Express, Discover
        - No additional processing fees
        - Instant payment confirmation
        
        ## Bank Account (ACH)
        - Checking and savings accounts accepted
        - No fees for ACH payments
        - Processing time: 1-2 business days
        - Preferred method for autopay
        
        ## Other Methods
        - Wire transfer: For large payments (fees may apply)
        - Check by mail: Allow 7-10 days processing
        - Payment by phone: Call support during business hours
        
        ## Security
        - All payments encrypted with 256-bit SSL
        - PCI-DSS Level 1 compliant
        - Tokenized card storage
        - Real-time fraud monitoring
        
        ## Updating Payment Methods
        - Add new method: Instant activation
        - Remove method: Takes effect after pending payments clear
        - Set default: Changes apply to future autopay
        """,
        metadata={
            "topic": "payment_methods",
            "category": "payments",
            "last_updated": "2024-08-01"
        }
    ),
    
    # Late Payments
    Document(
        page_content="""
        # Late Payment Policy
        
        We understand life happens. Here's what to know about late payments.
        
        ## Grace Period
        - 15-day grace period after due date
        - No late fees during grace period
        - No credit reporting during grace period
        
        ## Late Fees
        - After grace period: $25 or 5% of payment, whichever is less
        - Maximum one late fee per billing cycle
        - Late fees waived once per 12-month period upon request
        
        ## Credit Reporting
        - 30+ days late: May be reported to credit bureaus
        - Contact us before 30 days to arrange payment plan
        - Positive payment history reported monthly
        
        ## Hardship Assistance
        If experiencing financial hardship:
        1. Contact us immediately
        2. We offer temporary payment reductions
        3. Payment deferrals available (up to 3 months)
        4. Loan modification options for long-term hardship
        
        ## Reinstating Account
        - Pay past due amount to bring current
        - Autopay re-enrollment recommended
        - Account reinstated within 24 hours of payment
        """,
        metadata={
            "topic": "late_payments",
            "category": "policies",
            "last_updated": "2024-07-15"
        }
    )
]


# ============================================================================
# RAG Pipeline Class with Pinecone
# ============================================================================

class PaymentRAG:
    """RAG pipeline for payment knowledge retrieval using Pinecone."""
    
    def __init__(
        self,
        index_name: str = PINECONE_INDEX_NAME,
        namespace: str = PINECONE_NAMESPACE,
        api_key: Optional[str] = None
    ):
        self.index_name = index_name
        self.namespace = namespace
        self.api_key = api_key or os.getenv("PINECONE_API_KEY")
        
        # Initialize Pinecone client
        self.pc = None
        self.index = None
        
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(
            model=EMBEDDING_MODEL
        )
        
        # Initialize vector store
        self.vectorstore = None
        self._init_pinecone()
    
    def _init_pinecone(self):
        """Initialize Pinecone client and index."""
        
        if not self.api_key:
            print("⚠️  PINECONE_API_KEY not set. Running in mock mode.")
            return
        
        try:
            # Initialize Pinecone client
            self.pc = Pinecone(api_key=self.api_key)
            
            # Check if index exists
            existing_indexes = [idx.name for idx in self.pc.list_indexes()]
            
            if self.index_name not in existing_indexes:
                # Create new index
                print(f"Creating new Pinecone index: {self.index_name}")
                self.pc.create_index(
                    name=self.index_name,
                    dimension=EMBEDDING_DIMENSION,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud=PINECONE_CLOUD,
                        region=PINECONE_REGION
                    )
                )
                # Wait for index to be ready
                print("Waiting for index to be ready...")
                time.sleep(10)
            
            # Connect to index
            self.index = self.pc.Index(self.index_name)
            
            # Initialize LangChain Pinecone vectorstore
            self.vectorstore = PineconeVectorStore(
                index=self.index,
                embedding=self.embeddings,
                namespace=self.namespace
            )
            
            # Check if we need to index default documents
            stats = self.index.describe_index_stats()
            namespace_stats = stats.namespaces.get(self.namespace, {})
            vector_count = namespace_stats.get("vector_count", 0) if namespace_stats else 0
            
            if vector_count == 0:
                print("Index is empty. Indexing default documents...")
                self.index_documents(PAYMENT_KNOWLEDGE_DOCS)
            else:
                print(f"Connected to Pinecone index with {vector_count} vectors")
            
        except Exception as e:
            print(f"Error initializing Pinecone: {e}")
            print("Running in mock mode.")
    
    def _generate_doc_id(self, doc: Document, chunk_index: int) -> str:
        """Generate a unique ID for a document chunk."""
        content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()[:8]
        topic = doc.metadata.get("topic", "unknown")
        return f"{topic}_{content_hash}_{chunk_index}"
    
    def index_documents(self, documents: List[Document], batch_size: int = 100):
        """Index documents into Pinecone."""
        
        if not self.vectorstore:
            print("Pinecone not initialized. Cannot index documents.")
            return
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        chunks = text_splitter.split_documents(documents)
        print(f"Split {len(documents)} documents into {len(chunks)} chunks")
        
        # Generate IDs for chunks
        ids = [self._generate_doc_id(chunk, i) for i, chunk in enumerate(chunks)]
        
        # Add to Pinecone in batches
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            batch_ids = ids[i:i + batch_size]
            
            self.vectorstore.add_documents(
                documents=batch_chunks,
                ids=batch_ids
            )
            print(f"Indexed batch {i // batch_size + 1}/{(len(chunks) + batch_size - 1) // batch_size}")
        
        print(f"✅ Indexed {len(chunks)} chunks into Pinecone")
    
    def search(
        self,
        query: str,
        k: int = 4,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """Search for relevant documents."""
        
        if not self.vectorstore:
            return self._mock_search(query, k)
        
        # Perform similarity search
        if filter_metadata:
            results = self.vectorstore.similarity_search(
                query,
                k=k,
                filter=filter_metadata
            )
        else:
            results = self.vectorstore.similarity_search(
                query,
                k=k
            )
        
        return results
    
    def search_with_scores(
        self,
        query: str,
        k: int = 4,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[tuple[Document, float]]:
        """Search with relevance scores."""
        
        if not self.vectorstore:
            return self._mock_search_with_scores(query, k)
        
        results = self.vectorstore.similarity_search_with_score(
            query,
            k=k,
            filter=filter_metadata
        )
        
        # Pinecone returns (doc, score) where score is distance
        # Convert to relevance score (1 - distance for cosine)
        return [(doc, 1 - score) for doc, score in results]
    
    def hybrid_search(
        self,
        query: str,
        k: int = 4,
        alpha: float = 0.5
    ) -> List[Document]:
        """
        Hybrid search combining dense and sparse vectors.
        Alpha controls the weighting: 0 = sparse only, 1 = dense only.
        
        Note: Requires Pinecone index with hybrid search enabled.
        """
        if not self.vectorstore:
            return self._mock_search(query, k)
        
        # For basic implementation, use dense search
        # Full hybrid search requires sparse encoder (BM25 or SPLADE)
        return self.search(query, k)
    
    def search_by_topic(
        self,
        query: str,
        topic: str,
        k: int = 4
    ) -> List[Document]:
        """Search within a specific topic category."""
        
        return self.search(
            query,
            k=k,
            filter_metadata={"topic": {"$eq": topic}}
        )
    
    def get_retriever(self, k: int = 4, filter_metadata: Optional[Dict] = None):
        """Get a retriever for use in chains."""
        
        if not self.vectorstore:
            # Return mock retriever
            from langchain_core.retrievers import BaseRetriever
            
            class MockRetriever(BaseRetriever):
                def _get_relevant_documents(self_, query: str) -> List[Document]:
                    return self._mock_search(query, k)
            
            return MockRetriever()
        
        search_kwargs = {"k": k}
        if filter_metadata:
            search_kwargs["filter"] = filter_metadata
        
        return self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs=search_kwargs
        )
    
    def delete_namespace(self):
        """Delete all vectors in the current namespace."""
        if self.index:
            self.index.delete(delete_all=True, namespace=self.namespace)
            print(f"Deleted all vectors in namespace: {self.namespace}")
    
    def get_index_stats(self) -> Dict:
        """Get Pinecone index statistics."""
        if not self.index:
            return {"error": "Index not initialized"}
        
        stats = self.index.describe_index_stats()
        return {
            "total_vector_count": stats.total_vector_count,
            "dimension": stats.dimension,
            "namespaces": {
                ns: {"vector_count": info.vector_count}
                for ns, info in stats.namespaces.items()
            }
        }
    
    # ========================================================================
    # Mock Methods (for testing without Pinecone)
    # ========================================================================
    
    def _mock_search(self, query: str, k: int) -> List[Document]:
        """Mock search for testing without Pinecone."""
        
        # Simple keyword matching
        query_lower = query.lower()
        scored_docs = []
        
        for doc in PAYMENT_KNOWLEDGE_DOCS:
            score = 0
            content_lower = doc.page_content.lower()
            topic = doc.metadata.get("topic", "")
            
            # Score based on keyword matches
            keywords = query_lower.split()
            for keyword in keywords:
                if keyword in content_lower:
                    score += 1
                if keyword in topic:
                    score += 2
            
            if score > 0:
                scored_docs.append((doc, score))
        
        # Sort by score and return top k
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scored_docs[:k]]
    
    def _mock_search_with_scores(self, query: str, k: int) -> List[tuple[Document, float]]:
        """Mock search with scores for testing."""
        
        docs = self._mock_search(query, k)
        # Assign fake relevance scores
        return [(doc, 0.9 - i * 0.1) for i, doc in enumerate(docs)]


# ============================================================================
# RAG Chain with LLM
# ============================================================================

class PaymentRAGChain:
    """RAG chain that combines retrieval with LLM generation."""
    
    RAG_PROMPT = """You are a helpful payment assistant. Use the following context to answer the customer's question.
    If the context doesn't contain relevant information, say so and provide general guidance.
    
    Context:
    {context}
    
    Question: {question}
    
    Provide a helpful, accurate answer based on the context. If citing specific policies, mention the source.
    """
    
    def __init__(self, rag: Optional[PaymentRAG] = None):
        self.rag = rag or PaymentRAG()
        
        # Initialize LLM
        self.llm = ChatAnthropic(
            model="claude-sonnet-4-20250514",
            temperature=0.1,
            max_tokens=1024
        )
        
        # Create prompt
        self.prompt = ChatPromptTemplate.from_template(self.RAG_PROMPT)
        
        # Build chain
        self.chain = self._build_chain()
    
    def _build_chain(self):
        """Build the RAG chain."""
        
        retriever = self.rag.get_retriever(k=4)
        
        def format_docs(docs):
            return "\n\n".join([
                f"[{doc.metadata.get('topic', 'general')}]\n{doc.page_content}"
                for doc in docs
            ])
        
        chain = (
            {
                "context": retriever | format_docs,
                "question": RunnablePassthrough()
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        
        return chain
    
    def query(self, question: str) -> dict:
        """Query the RAG chain and return response with sources."""
        
        # Get retrieved documents
        docs = self.rag.search(question, k=4)
        
        # Generate response
        response = self.chain.invoke(question)
        
        return {
            "question": question,
            "answer": response,
            "sources": [
                {
                    "topic": doc.metadata.get("topic"),
                    "category": doc.metadata.get("category"),
                    "content_preview": doc.page_content[:200] + "..."
                }
                for doc in docs
            ],
            "timestamp": datetime.now().isoformat()
        }


# ============================================================================
# Demo / Test Functions
# ============================================================================

def demo_rag():
    """Demo the RAG pipeline with Pinecone."""
    
    print("=" * 60)
    print("Payment RAG Pipeline Demo (Pinecone)")
    print("=" * 60)
    
    # Check for API keys
    has_openai = bool(os.getenv("OPENAI_API_KEY"))
    has_pinecone = bool(os.getenv("PINECONE_API_KEY"))
    
    if not has_openai:
        print("\n⚠️  OPENAI_API_KEY not set.")
    if not has_pinecone:
        print("\n⚠️  PINECONE_API_KEY not set.")
    
    if not has_openai or not has_pinecone:
        print("\nRunning in mock mode...\n")
    
    # Initialize RAG
    print("\nInitializing RAG pipeline with Pinecone...")
    rag = PaymentRAG()
    
    # Show index stats if available
    stats = rag.get_index_stats()
    if "error" not in stats:
        print(f"\n📊 Pinecone Index Stats:")
        print(f"   Total vectors: {stats.get('total_vector_count', 'N/A')}")
        print(f"   Dimension: {stats.get('dimension', 'N/A')}")
        if stats.get('namespaces'):
            for ns, info in stats['namespaces'].items():
                print(f"   Namespace '{ns}': {info.get('vector_count', 0)} vectors")
    
    # Test queries
    test_queries = [
        "What is your refund policy?",
        "How do I set up autopay?",
        "What happens if I miss a payment?",
        "Can I pay off my solar loan early?",
        "What payment methods do you accept?"
    ]
    
    print("\n" + "-" * 60)
    print("Testing semantic search:")
    print("-" * 60)
    
    for query in test_queries:
        print(f"\n📝 Query: {query}")
        results = rag.search_with_scores(query, k=2)
        
        for doc, score in results:
            print(f"   └─ [{score:.3f}] {doc.metadata.get('topic')}: {doc.page_content[:100]}...")
    
    # Test topic-filtered search
    print("\n" + "-" * 60)
    print("Testing topic-filtered search:")
    print("-" * 60)
    
    print(f"\n📝 Query: 'How long does it take?' (topic: refund_policy)")
    results = rag.search_by_topic("How long does it take?", topic="refund_policy", k=2)
    for doc in results:
        print(f"   └─ {doc.metadata.get('topic')}: {doc.page_content[:100]}...")
    
    # Test RAG chain if APIs available
    if has_openai and has_pinecone:
        print("\n" + "-" * 60)
        print("Testing RAG chain with LLM:")
        print("-" * 60)
        
        try:
            chain = PaymentRAGChain(rag)
            
            result = chain.query("I was charged twice, what should I do?")
            print(f"\n📝 Question: {result['question']}")
            print(f"\n🤖 Answer: {result['answer']}")
            print(f"\n📚 Sources: {[s['topic'] for s in result['sources']]}")
        except Exception as e:
            print(f"\n⚠️  RAG chain error: {e}")
    else:
        print("\n" + "-" * 60)
        print("Skipping RAG chain test (requires API keys)")
        print("-" * 60)


def setup_pinecone_index():
    """Utility function to set up Pinecone index with sample documents."""
    
    print("Setting up Pinecone index...")
    
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        print("❌ PINECONE_API_KEY not set")
        return
    
    rag = PaymentRAG()
    
    # Delete existing vectors and re-index
    print("Clearing existing vectors...")
    rag.delete_namespace()
    
    print("Indexing sample documents...")
    rag.index_documents(PAYMENT_KNOWLEDGE_DOCS)
    
    print("✅ Setup complete!")
    print(rag.get_index_stats())


if __name__ == "__main__":
    demo_rag()
