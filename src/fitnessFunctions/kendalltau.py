"""
Kendall Tau distance between two rankings
"""

from scipy.stats import kendalltau


def kendalltau_distance(ranking1, ranking2):
    """
    Computes the Kendall Tau distance between two rankings
    """
    return kendalltau(ranking1, ranking2).statistic
