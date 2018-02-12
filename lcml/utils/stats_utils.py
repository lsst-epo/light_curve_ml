from scipy import stats


def confidenceInterval(values, mean, confidence=0.99):
    """Calculates confidence interval using Student's t-distribution and
    standard error of mean"""
    return stats.t.interval(confidence, len(values) - 1, loc=mean,
                            scale=stats.sem(values))

