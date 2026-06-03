"""
Spearman's r correlation between two rankings
"""

from scipy.stats import spearmanr


def spearmanr_correlation(ranking1, ranking2):
    """
    Computes the Spearman's r correlation between two rankings
    """
    return spearmanr(ranking1, ranking2).correlation
