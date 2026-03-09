from sklearn.utils.estimator_checks import check_estimator
from pygam import LinearGAM, LogisticGAM


estimators = [
    LinearGAM(),
    LogisticGAM()
]


for est in estimators:
    print("checking:", est.__class__.__name__)
    try:
        check_estimator(est)
    except Exception as e:
        print("failed:", type(e).__name__)
        print(e)
        print("-" * 40)