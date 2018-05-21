from collections import Counter
import logging
import operator
from typing import List

import numpy as np
from prettytable import PrettyTable

from lcml.utils.format_util import fmtPct


logger = logging.getLogger(__name__)


def fixedValueImpute(features: List[np.ndarray], value: float):
    """Sets non-finite feature values to specified value in-place"""
    imputes = Counter()
    vectorLengths = dict()
    for fv in features:
        vectorLengths[len(fv)] = fv
        if len(vectorLengths) > 1:
            raise ValueError("Multiple vector lengths: %s. Vectors: %s " % (
                list(vectorLengths), list(vectorLengths.values())))

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

    # Nice-to-have, associated Counter keys (int) with feature names:
    # ['Amplitude' 'AndersonDarling' 'Autocor_length' 'Beyond1Std' 'CAR_mean'
    #  'CAR_sigma' 'CAR_tau' 'Con' 'Eta_e' 'FluxPercentileRatioMid20'
    #  'FluxPercentileRatioMid35' 'FluxPercentileRatioMid50'
    #  'FluxPercentileRatioMid65' 'FluxPercentileRatioMid80'
    #  'Freq1_harmonics_amplitude_0' 'Freq1_harmonics_amplitude_1'
    #  'Freq1_harmonics_amplitude_2' 'Freq1_harmonics_amplitude_3'
    #  'Freq1_harmonics_rel_phase_0' 'Freq1_harmonics_rel_phase_1'
    #  'Freq1_harmonics_rel_phase_2' 'Freq1_harmonics_rel_phase_3'
    #  'Freq2_harmonics_amplitude_0' 'Freq2_harmonics_amplitude_1'
    #  'Freq2_harmonics_amplitude_2' 'Freq2_harmonics_amplitude_3'
    #  'Freq2_harmonics_rel_phase_0' 'Freq2_harmonics_rel_phase_1'
    #  'Freq2_harmonics_rel_phase_2' 'Freq2_harmonics_rel_phase_3'
    #  'Freq3_harmonics_amplitude_0' 'Freq3_harmonics_amplitude_1'
    #  'Freq3_harmonics_amplitude_2' 'Freq3_harmonics_amplitude_3'
    #  'Freq3_harmonics_rel_phase_0' 'Freq3_harmonics_rel_phase_1'
    #  'Freq3_harmonics_rel_phase_2' 'Freq3_harmonics_rel_phase_3' 'Gskew'
    #  'LinearTrend' 'MaxSlope' 'Mean' 'Meanvariance' 'MedianAbsDev' 'MedianBRP'
    #  'PairSlopeTrend' 'PercentAmplitude' 'PercentDifferenceFluxPercentile'
    #  'PeriodLS' 'Period_fit' 'Psi_CS' 'Psi_eta' 'Q31' 'Rcs' 'Skew'
    #  'SlottedA_length' 'SmallKurtosis' 'Std' 'StetsonK' 'StetsonK_AC'
    #  'StructureFunction_index_21' 'StructureFunction_index_31'
    #  'StructureFunction_index_32']
