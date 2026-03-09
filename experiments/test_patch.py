import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import r2_score, make_scorer
from sklearn.model_selection import KFold

from patches.sklearn_tags_patch import PatchedGAM


X = np.random.randn(100,3)
y = 2*X[:,0] + np.random.randn(100)

model = PatchedGAM()

search = RandomizedSearchCV(
    model,
    cv=KFold(3),
    param_distributions={"max_iter":[50,100]},
    n_iter=2,
    scoring=make_scorer(r2_score),
)

search.fit(X,y)

print("search completed")