from collections import Counter
import operator
from typing import List

import numpy as np
from prettytable import PrettyTable

from lcml.utils.basic_logging import BasicLogging
from lcml.utils.format_util import fmtPct


logger = BasicLogging.getLogger(__name__)


def fixedValueImpute(features: List[List[float]], value: float):
    """Sets non-finite feature values to specified value"""
    imputes = Counter()
    for fv in features:
        for i, v in enumerate(fv):
            if not np.isfinite(v):
                imputes[i] += 1
                fv[i] = value

    if imputes:
        t = PrettyTable(["feature", "imputes", "impute rate",
                         "percentage of all imputes"])
        vectorCount = len(features)
        totalImputes = sum(imputes.values())
        for name, count in sorted(imputes.items(),
                                  key=operator.itemgetter(1),
                                  reverse=True):
            t.add_row([name, count, fmtPct(count, vectorCount),
                       fmtPct(count, totalImputes)])

        logger.info("\n" + str(t))
