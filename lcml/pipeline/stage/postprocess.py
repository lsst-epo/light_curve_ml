from typing import List

import numpy as np
from sklearn.preprocessing import StandardScaler

from lcml.pipeline.stage.feature_process import fixedValueImpute


def postprocessFeatures(features: List[np.ndarray], params: dict) -> np.ndarray:
    if params.get("impute", None):
        fixedValueImpute(features, value=0.0)
    if params.get("standardize", None):
        features = StandardScaler().fit_transform(features)

    return features
