#: Research by Kim has uncovered that light curves should have at least 80
#: data points to be classifiable
SUFFICIENT_LC_DATA = 80


def removeMachoOutliers(mjds, values, errors, remove=-99.0):
    """Simple bogus value filter for MACHO magnitudes and errors."""
    return zip(*[(mjds[i], v, errors[i])
                 for i, v in enumerate(values)
                 if v != remove and errors[i] != remove])
