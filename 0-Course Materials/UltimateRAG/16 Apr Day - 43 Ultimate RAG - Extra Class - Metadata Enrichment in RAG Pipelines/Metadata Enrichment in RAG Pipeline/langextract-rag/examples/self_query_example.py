#!/usr/bin/env python3
"""
Self-Query Retriever Example

Demonstrates how the SelfQueryRetriever automatically extracts metadata
filters from natural language queries.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import validate_config
from retrieval.retriever import get_self_query_retriever, print_search_results
from retrieval.rag_chain import create_self_query_rag_chain


def demonstrate_self_query():
    """Demonstrate SelfQueryRetriever capabilities."""
    
    # Validate configuration
    missing = validate_config()
    if missing:
        print("❌ Missing required configuration:")
        for item in missing:
            print(f"   - {item}")
        sys.exit(1)
    
    print("=" * 70)
    print("🤖 SELF-QUERY RETRIEVER DEMO")
    print("=" * 70)
    print("\nThe SelfQueryRetriever uses an LLM to automatically extract")
    print("metadata filters from your natural language queries.")
    print("-" * 70)
    
    # Example queries that demonstrate automatic filter extraction
    example_queries = [
        "What are the API rate limits in version 2.1?",
        "Show me technical documentation about OAuth2 authentication",
        "What did the Platform Engineering team say about API keys?",
        "Find financial documents from January 2025",
        "What are the legal compliance requirements?",
    ]
    
    print("\n📋 Example Queries and Expected Filters:")
    print("-" * 70)
    
    expected_filters = [
        {'version': 'v2.1', 'topic': 'API rate limits'},
        {'category': 'technical', 'topic': 'OAuth2 authentication'},
        {'entities': 'Platform Engineering', 'topic': 'API keys'},
        {'category': 'financial', 'release_date': 'January 2025'},
        {'category': 'legal', 'topic': 'compliance'},
    ]
    
    for i, (query, expected) in enumerate(zip(example_queries, expected_filters), 1):
        print(f"\n{i}. Query: \"{query}\"")
        print(f"   Expected filter: {expected}")
    
    print("\n" + "=" * 70)
    print("💡 How to use in your code:")
    print("=" * 70)
    
    code_example = '''
from retrieval.rag_chain import create_self_query_rag_chain

# Create the chain
chain = create_self_query_rag_chain(verbose=True)

# Ask a question - the LLM automatically generates filters!
response = chain.invoke(
    "What authentication methods are available in API version 2.1?"
)

# The LLM will automatically apply filters like:
# {"version": "v2.1", "topic": "Authentication API"}
'''
    print(code_example)
    
    print("\n" + "=" * 70)
    print("🔍 Try it yourself:")
    print("=" * 70)
    print("\nRun: python main.py query 'Your question' --self-query")
    print("\nOr use interactive mode:")
    print("  python main.py interactive")
    print("  Then type: /sq  (to enable SelfQueryRetriever)")


if __name__ == "__main__":
    demonstrate_self_query()
