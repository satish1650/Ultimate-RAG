"""
Evaluation Module

Provides tools for evaluating and comparing retrieval performance.
"""

from evaluation.visualize import (
    save_extraction_results_jsonl,
    generate_html_visualization,
    debug_document_metadata,
    analyze_metadata_coverage,
    print_metadata_analysis,
    export_metadata_csv,
    debug_retrieval_results,
    compare_metadata_distributions
)

__all__ = [
    "save_extraction_results_jsonl",
    "generate_html_visualization",
    "debug_document_metadata",
    "analyze_metadata_coverage",
    "print_metadata_analysis",
    "export_metadata_csv",
    "debug_retrieval_results",
    "compare_metadata_distributions"
]
