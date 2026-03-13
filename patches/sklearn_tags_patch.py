from sklearn.base import BaseEstimator
from pygam.pygam import GAM
from pygam.pygam import LogisticGAM


class PatchedGAM(GAM, BaseEstimator):

    # minimal sklearn tag support
    def __sklearn_tags__(self):
        return {
            "requires_y": True,
            "allow_nan": False,
            "non_deterministic": False,
            "estimator_type": "regressor",
        }


class PatchedLogisticGAM(LogisticGAM, BaseEstimator):

    def __sklearn_tags__(self):
        return {
            "requires_y": True,
            "allow_nan": False,
            "non_deterministic": False,
            "estimator_type": "classifier",
        }