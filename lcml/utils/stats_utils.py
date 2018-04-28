from scipy import stats


def confidenceInterval(values, mean: float, confidence: float=0.99) -> (float,
                                                                        float):
    """Calculates confidence interval using Student's t-distribution and
    standard error of mean"""
    return stats.t.interval(confidence, len(values) - 1, loc=mean,
                            scale=stats.sem(values))
