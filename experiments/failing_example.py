import numpy as np
import sklearn
from pygam import GAM
from sklearn.metrics import r2_score, make_scorer
from sklearn.model_selection import KFold, RandomizedSearchCV

print("sklearn version:", sklearn.__version__)

np.random.seed(0)
X = np.random.randn(120, 3)
y = 2 * X[:, 0] + 0.5 * X[:, 1] + np.random.randn(120)

model = GAM()
print("estimator:", model)

scorer = make_scorer(r2_score, greater_is_better=True)

search = RandomizedSearchCV(
    model,
    cv=KFold(3),
    param_distributions={"max_iter": [50, 100]},
    n_iter=2,
    scoring=scorer,
)

try:
    search.fit(X, y)
except Exception as e:
    print("error during search:")
    print(type(e).__name__, e)