```python
# quick experiment: making GAM inherit from sklearn BaseEstimator
# just testing if this helps with sklearn>=1.7 tools


from sklearn.base import BaseEstimator
from pygam.pygam import GAM
from pygam.pygam import LogisticGAM


class GAMWithBase(BaseEstimator, GAM):
    pass


class LogisticGAMWithBase(BaseEstimator, LogisticGAM):
    pass


if __name__ == "__main__":

    import numpy as np
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.model_selection import KFold
    from sklearn.metrics import r2_score, make_scorer

    X = np.random.randn(120, 3)
    y = X[:, 0] * 2 + np.random.randn(120)

    model = GAMWithBase()

    search = RandomizedSearchCV(
        model,
        cv=KFold(3),
        param_distributions={"max_iter": [50, 100]},
        n_iter=2,
        scoring=make_scorer(r2_score),
    )

    try:
        search.fit(X, y)
        print("search finished")
    except Exception as e:
        print("still failing")
        print(type(e).__name__, e)