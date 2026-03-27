"""
BayesDesign: Bayesian protein sequence design using language models and structural constraints.
"""

__version__ = "0.1.0"

# Import main functions for convenience
from bayes_design.design import design_seqs
from bayes_design.score_sequences import score_sequences

__all__ = [
    "design_seqs",
    "score_sequences",
]
