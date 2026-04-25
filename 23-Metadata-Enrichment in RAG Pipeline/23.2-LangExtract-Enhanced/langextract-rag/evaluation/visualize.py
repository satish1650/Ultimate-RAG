"""
Visualization and Debugging Utilities

Provides tools for visualizing LangExtract results and debugging the RAG pipeline.
"""

import json
import textwrap
from pathlib import Path
from typing import Dict, List, Optional, Any

import langextract as lx
from langchain_core.documents import Document

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# LangExtract Visualization
# =============================================================================

def save_extraction_results_jsonl(
    documents: List[Document],
    output_path: str | Path,
    include_full_text: bool = True
):
    """
    Save LangExtract results to a JSONL file for visualization.
    
    Args:
        documents: List of enriched Document objects
        output_path: Path to save the JSONL file
        include_full_text: Whether to include full text content
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    records = []
    for i, doc in enumerate(documents):
        record = {
            "id": f"doc_{i}",
            "metadata": doc.metadata
        }
        if include_full_text:
            record["text"] = doc.page_content
        records.append(record)
    
    with open(output_path, 'w') as f:
        for record in records:
            f.write(json.dumps(record) + '\n')
    
    print(f"💾 Saved {len(records)} records to {output_path}")


def generate_html_visualization(
    jsonl_path: str | Path,
    output_html_path: Optional[str | Path] = None
) -> str:
    """
    Generate an interactive HTML visualization from extraction results.
    
    Note: This is a simplified visualization. For full LangExtract visualization,
    use lx.visualize() with proper extraction results.
    
    Args:
        jsonl_path: Path to the JSONL file
        output_html_path: Path to save the HTML file
        
    Returns:
        HTML content as string
    """
    jsonl_path = Path(jsonl_path)
    
    if not jsonl_path.exists():
        raise FileNotFoundError(f"File not found: {jsonl_path}")
    
    # Load records
    records = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            records.append(json.loads(line.strip()))
    
    # Generate simple HTML visualization
    html_content = _generate_simple_html(records)
    
    if output_html_path:
        output_html_path = Path(output_html_path)
        output_html_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_html_path, 'w') as f:
            f.write(html_content)
        print(f"🌐 Visualization saved to {output_html_path}")
    
    return html_content


def _generate_simple_html(records: List[Dict]) -> str:
    """Generate a simple HTML visualization for document metadata."""
    
    html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LangExtract RAG - Metadata Visualization</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        h1 {
            color: white;
            text-align: center;
            margin-bottom: 30px;
            font-size: 2.5em;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        .stats {
            background: rgba(255,255,255,0.95);
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
        }
        .stats h2 {
            color: #333;
            margin-bottom: 15px;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }
        .stat-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
        }
        .stat-value {
            font-size: 2em;
            font-weight: bold;
        }
        .stat-label {
            font-size: 0.9em;
            opacity: 0.9;
        }
        .document-card {
            background: rgba(255,255,255,0.95);
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            transition: transform 0.2s;
        }
        .document-card:hover {
            transform: translateY(-5px);
        }
        .doc-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 2px solid #eee;
        }
        .doc-id {
            font-weight: bold;
            color: #667eea;
            font-size: 1.2em;
        }
        .doc-source {
            color: #666;
            font-size: 0.9em;
        }
        .metadata-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
            margin-bottom: 15px;
        }
        .metadata-item {
            background: #f8f9fa;
            padding: 10px;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }
        .metadata-label {
            font-size: 0.8em;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .metadata-value {
            font-size: 1em;
            color: #333;
            font-weight: 500;
            margin-top: 3px;
        }
        .doc-content {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
            line-height: 1.5;
            max-height: 200px;
            overflow-y: auto;
            color: #333;
        }
        .category-badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.8em;
            font-weight: 500;
            text-transform: uppercase;
        }
        .category-technical { background: #e3f2fd; color: #1976d2; }
        .category-legal { background: #fce4ec; color: #c2185b; }
        .category-financial { background: #e8f5e9; color: #388e3c; }
        .category-medical { background: #fff3e0; color: #f57c00; }
        .category-general { background: #f3e5f5; color: #7b1fa2; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔍 LangExtract-Enhanced RAG</h1>
        <h1 style="font-size: 1.5em; margin-top: -20px;">Document Metadata Visualization</h1>
        
        <div class="stats">
            <h2>📊 Overview</h2>
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-value">""" + str(len(records)) + """</div>
                    <div class="stat-label">Total Documents</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">""" + str(len(set(r['metadata'].get('category', 'unknown') for r in records))) + """</div>
                    <div class="stat-label">Categories</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">""" + str(len(set(r['metadata'].get('source', 'unknown') for r in records))) + """</div>
                    <div class="stat-label">Sources</div>
                </div>
            </div>
        </div>
"""
    
    # Add document cards
    for i, record in enumerate(records):
        metadata = record.get('metadata', {})
        content = record.get('text', '')[:500] + '...' if len(record.get('text', '')) > 500 else record.get('text', '')
        
        category = metadata.get('category', 'general')
        category_class = f'category-{category}' if category in ['technical', 'legal', 'financial', 'medical', 'general'] else 'category-general'
        
        html += f"""
        <div class="document-card">
            <div class="doc-header">
                <span class="doc-id">Document {i + 1}</span>
                <span class="category-badge {category_class}">{category}</span>
            </div>
            <div class="metadata-grid">
                <div class="metadata-item">
                    <div class="metadata-label">Source</div>
                    <div class="metadata-value">{metadata.get('source', 'N/A')}</div>
                </div>
                <div class="metadata-item">
                    <div class="metadata-label">Topic</div>
                    <div class="metadata-value">{metadata.get('topic', 'N/A')}</div>
                </div>
                <div class="metadata-item">
                    <div class="metadata-label">Version</div>
                    <div class="metadata-value">{metadata.get('version', 'N/A')}</div>
                </div>
                <div class="metadata-item">
                    <div class="metadata-label">Entities</div>
                    <div class="metadata-value">{metadata.get('entities', 'N/A')[:50]}...</div>
                </div>
            </div>
            <div class="doc-content">{content}</div>
        </div>
"""
    
    html += """
    </div>
</body>
</html>
"""
    
    return html


# =============================================================================
# Debugging Utilities
# =============================================================================

def debug_document_metadata(documents: List[Document], n: int = 5):
    """
    Print detailed metadata for debugging.
    
    Args:
        documents: List of Document objects
        n: Number of documents to print
    """
    print("\n" + "=" * 70)
    print("🔍 DEBUG: Document Metadata")
    print("=" * 70)
    
    for i, doc in enumerate(documents[:n]):
        print(f"\n📄 Document {i + 1}:")
        print("-" * 50)
        
        # Print all metadata
        for key, value in sorted(doc.metadata.items()):
            value_str = str(value)
            if len(value_str) > 100:
                value_str = value_str[:100] + "..."
            print(f"  {key}: {value_str}")
        
        # Print content preview
        content_preview = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
        print(f"\n  Content Preview:")
        print(f"  {content_preview}")


def analyze_metadata_coverage(documents: List[Document]) -> Dict[str, Any]:
    """
    Analyze the coverage of metadata fields across documents.
    
    Args:
        documents: List of Document objects
        
    Returns:
        Dictionary with coverage statistics
    """
    if not documents:
        return {}
    
    # Collect all metadata keys
    all_keys = set()
    for doc in documents:
        all_keys.update(doc.metadata.keys())
    
    # Calculate coverage for each key
    coverage = {}
    for key in sorted(all_keys):
        count = sum(1 for doc in documents if doc.metadata.get(key))
        coverage[key] = {
            "count": count,
            "percentage": (count / len(documents)) * 100
        }
    
    return coverage


def print_metadata_analysis(documents: List[Document]):
    """
    Print metadata coverage analysis.
    
    Args:
        documents: List of Document objects
    """
    coverage = analyze_metadata_coverage(documents)
    
    print("\n" + "=" * 70)
    print("📊 METADATA COVERAGE ANALYSIS")
    print("=" * 70)
    print(f"Total documents: {len(documents)}\n")
    
    print(f"{'Field':<20} {'Count':<10} {'Coverage':<10}")
    print("-" * 50)
    
    for field, stats in sorted(coverage.items(), key=lambda x: -x[1]["percentage"]):
        bar = "█" * int(stats["percentage"] / 10)
        print(f"{field:<20} {stats['count']:<10} {stats['percentage']:>5.1f}% {bar}")


def export_metadata_csv(
    documents: List[Document],
    output_path: str | Path
):
    """
    Export metadata to CSV for analysis in spreadsheet tools.
    
    Args:
        documents: List of Document objects
        output_path: Path to save the CSV file
    """
    import csv
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Collect all metadata keys
    all_keys = set()
    for doc in documents:
        all_keys.update(doc.metadata.keys())
    
    fieldnames = ['index', 'content_preview'] + sorted(all_keys)
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for i, doc in enumerate(documents):
            row = {
                'index': i,
                'content_preview': doc.page_content[:200].replace('\n', ' ')
            }
            row.update(doc.metadata)
            writer.writerow(row)
    
    print(f"📄 Metadata exported to {output_path}")


# =============================================================================
# Retrieval Debugging
# =============================================================================

def debug_retrieval_results(
    query: str,
    results: List[Any],
    method: str = "unknown"
):
    """
    Print detailed retrieval results for debugging.
    
    Args:
        query: The search query
        results: List of results (documents or (doc, score) tuples)
        method: Retrieval method name
    """
    print("\n" + "=" * 70)
    print(f"🔍 DEBUG: Retrieval Results ({method})")
    print("=" * 70)
    print(f"Query: {query}")
    print(f"Results: {len(results)}\n")
    
    for i, item in enumerate(results):
        # Handle both (doc, score) tuples and doc objects
        if isinstance(item, tuple):
            doc, score = item
            print(f"Result {i + 1} (Score: {score:.3f}):")
        else:
            doc = item
            print(f"Result {i + 1}:")
        
        print(f"  Source: {doc.metadata.get('source', 'N/A')}")
        print(f"  Page: {doc.metadata.get('page', 'N/A')}")
        print(f"  Topic: {doc.metadata.get('topic', 'N/A')}")
        print(f"  Category: {doc.metadata.get('category', 'N/A')}")
        print(f"  Preview: {doc.page_content[:100]}...")
        print()


def compare_metadata_distributions(
    documents_before: List[Document],
    documents_after: List[Document]
):
    """
    Compare metadata distributions before and after LangExtract enrichment.
    
    Args:
        documents_before: Documents before enrichment
        documents_after: Documents after enrichment
    """
    print("\n" + "=" * 70)
    print("📊 METADATA ENRICHMENT COMPARISON")
    print("=" * 70)
    
    # Before enrichment
    before_keys = set()
    for doc in documents_before:
        before_keys.update(doc.metadata.keys())
    
    # After enrichment
    after_keys = set()
    for doc in documents_after:
        after_keys.update(doc.metadata.keys())
    
    new_keys = after_keys - before_keys
    
    print(f"\nDocuments: {len(documents_before)} → {len(documents_after)}")
    print(f"Metadata fields: {len(before_keys)} → {len(after_keys)}")
    print(f"\nNew fields added by LangExtract:")
    for key in sorted(new_keys):
        print(f"  ✓ {key}")


# =============================================================================
# Module Entry Point
# =============================================================================

if __name__ == "__main__":
    print("Visualization and Debugging Utilities")
    print("-" * 60)
    print("\nAvailable functions:")
    print("  - save_extraction_results_jsonl()")
    print("  - generate_html_visualization()")
    print("  - debug_document_metadata()")
    print("  - analyze_metadata_coverage()")
    print("  - print_metadata_analysis()")
    print("  - export_metadata_csv()")
    print("  - debug_retrieval_results()")
    print("  - compare_metadata_distributions()")
