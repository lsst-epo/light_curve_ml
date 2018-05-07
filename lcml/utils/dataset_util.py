import numpy as np
from prettytable import PrettyTable
from typing import Dict, List, Tuple

from lcml.utils.basic_logging import BasicLogging
from lcml.utils.format_util import fmtPct


logger = BasicLogging.getLogger(__name__)


def convertClassLabels(labels: List[str]) -> (List[int], Dict[int, str]):
    """Converts all class labels in-place to integer values for each unique
    class. Also returns a decoder mapping from integer to readable label
    :param labels: class labels of some dataset
    :return Labels converted to ints as well as a mapping from integer class
    label to original string class label
    """
    # 'LPV'-> 1
    labelToInt = {v: i for i, v in enumerate(np.unique(labels))}
    for i in range(len(labels)):
        labels[i] = labelToInt[labels[i]]

    # 1 -> 'LPV'
    intToLabel = {i: v for v, i in labelToInt.items()}
    return labels, intToLabel


def reportDataset(dataset: list, labels: list=None):
    """Reports the characteristics of a dataset"""
    size = len(dataset)
    dataSizes = [len(x) for x in dataset]
    minSize = min(dataSizes)
    maxSize = max(dataSizes)
    ave = np.average(dataSizes)
    std = float(np.std(dataSizes))
    print("_Dataset Report_")
    print("size: %s \nmin: %s \nave: %.02f (%.02f) \nmax: %s" % (
        size, minSize, ave, std, maxSize))
    if labels:
        print("Unique labels: %s" % sorted(np.unique(labels)))


def attachLabels(values: List[float],
                 indexToLabel: Dict[int, str]) -> List[Tuple[str, float]]:
    """Attaches readable labels to a list of values.

    :param values: a list of object to be labeled
    :param indexToLabel: a mapping from index (int) to label (string
    :return list of two-tuples containing label and score
    """
    return [(indexToLabel[i], v) for i, v in enumerate(values)]


def reportClassHistogram(labels: Dict[str, int]):
    """Logs a histogram of the distribution of class labels
    :param labels: dict from label to frequency
    """
    t = PrettyTable(["category", "count", "percentage"])
    t.align = "l"
    total = sum(labels.values())
    for k, v in sorted(labels.items(), key=lambda x: x[1], reverse=True):
        t.add_row([k, v, fmtPct(v, total)])

    logger.info("\n" + str(t))