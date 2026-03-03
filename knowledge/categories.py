"""
Knowledge Category Definitions
================================
Default categories and metadata for the knowledge tree.
"""

DEFAULT_CATEGORIES = {
    "papers": {
        "description": "Academic papers, preprints, and literature reviews",
        "icon": "paper",
        "extensions": [".md", ".json", ".bib"],
    },
    "experiments": {
        "description": "Experiment results, benchmarks, and evaluations",
        "icon": "flask",
        "extensions": [".md", ".json", ".csv"],
    },
    "methods": {
        "description": "Algorithms, techniques, and methodologies",
        "icon": "gear",
        "extensions": [".md", ".json"],
    },
    "code": {
        "description": "Code implementations, scripts, and snippets",
        "icon": "code",
        "extensions": [".py", ".md", ".json"],
    },
    "reports": {
        "description": "Progress reports and summaries",
        "icon": "report",
        "extensions": [".md"],
    },
}

# Max items per directory before triggering auto-reorganization
REORG_THRESHOLD = 20

# Target number of subcategories when splitting
SPLIT_TARGET = 4
