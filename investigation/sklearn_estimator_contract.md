# sklearn estimator contract (notes)

These are my notes while trying to understand what **scikit-learn expects from an estimator**.

I started looking at this because `pyGAM` crashes when used with `RandomizedSearchCV` in newer versions of sklearn (>=1.7).

The error mentions something about:

```id="n0j9kq"
__sklearn_tags__()
```

so I tried to understand what the estimator requirements actually are.

---

## what is an estimator in sklearn ?

In sklearn basically **everything is an estimator**.

Examples:

* LinearRegression
* RandomForestClassifier
* SVC
* PCA
* KMeans

They all follow the same interface.

Typical usage looks like:

```id="pghj9n"
model = SomeEstimator(param=value)
model.fit(X, y)
pred = model.predict(X)
```

So sklearn tools (pipeline, gridsearch etc) assume this pattern.

---

## the BaseEstimator class

Most sklearn estimators inherit from

```id="rjz71v"
sklearn.base.BaseEstimator
```

This class provides some important functionality automatically.

Main things it provides:

* parameter inspection
* get_params()
* set_params()
* clone() support
* estimator tags (I think)

Example simplified:

```id="q13x2f"
class MyEstimator(BaseEstimator):

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        ...

    def predict(self, X):
        ...
```

Important thing: **all parameters must be in the constructor**.

---

## get_params()

`get_params()` returns the estimator parameters.

Example:

```id="xru5o7"
model = LinearRegression(fit_intercept=True)
model.get_params()
```

returns something like

```id="w2ny2b"
{
 'fit_intercept': True,
 'copy_X': True,
 'n_jobs': None
}
```

This is used internally by things like

* GridSearchCV
* RandomizedSearchCV
* Pipeline

The key idea is that sklearn must be able to **inspect parameters programmatically**.

---

## set_params()

This is used when hyperparameters are modified.

Example:

```id="q0e9pa"
model.set_params(alpha=0.1)
```

But it also supports **nested parameters**.

Example:

```id="72s8q0"
pipeline.set_params(model__alpha=0.1)
```

This is important for pipelines.

---

## clone()

`clone()` creates a fresh estimator with the same parameters but without fitted state.

Example:

```id="p7oex0"
from sklearn.base import clone

model2 = clone(model)
```

Important rule:

clone works only if **all parameters are stored as constructor arguments**.

This is why sklearn discourages things like mutable defaults.

Bad example:

```id="7e0fyo"
def __init__(self, callbacks=[]):
```

because the same list might get reused.

---

## estimator tags

This part seems newer.

In sklearn >=1.7 they introduced something called **estimator tags system**.

Tags describe properties of the estimator.

Examples:

* does it require y
* does it support multioutput
* classifier vs regressor
* supports sparse input

Tags help sklearn tools behave correctly.

---

## **sklearn_tags**()

Newer sklearn versions expect estimators to implement:

```id="6ehywv"
__sklearn_tags__()
```

This method returns a **Tags object** describing the estimator.

If the estimator inherits from `BaseEstimator` this might already be implemented.

But if the class hierarchy does not include BaseEstimator, sklearn introspection may fail.

---

## example tag usage

Example inside sklearn (simplified):

```id="0ktyx2"
def get_tags(estimator):
    return estimator.__sklearn_tags__()
```

If this method doesn't exist → error.

Which is exactly what happens with pygam.

---

## estimator types

sklearn also distinguishes estimator types.

Examples:

regressors

* LinearRegression
* Ridge
* Lasso

classifiers

* LogisticRegression
* SVC
* RandomForestClassifier

These sometimes require different tags.

For example classifiers must support:

```id="y63c1k"
predict_proba()
```

or

```id="6sx6op"
decision_function()
```

---

## score() behavior

Another expectation in sklearn is the `score()` method.

Typical behavior:

Regressor

```id="x2xy6d"
score() -> R²
```

Classifier

```id="x9b7d1"
score() -> accuracy
```

However pygam seems to return **explained deviance** instead.

Not sure if that could break something in sklearn tools.

---

## sklearn meta estimators

Many sklearn utilities rely heavily on the estimator contract.

Examples:

* Pipeline
* GridSearchCV
* RandomizedSearchCV
* cross_val_score

These tools rely on

* get_params
* clone
* estimator tags
* fit/predict signatures

If any of these are missing the estimator might break.

---

## pygam situation

Looking at pygam code:

```id="s5j9zv"
class GAM(Core, MetaTermMixin):
```

It does **not inherit from BaseEstimator**.

Instead it has a custom `Core` class.

Core implements its own versions of

* get_params()
* set_params()

But they are not exactly the same as sklearn's.

Example:

```id="o5h5h7"
Core.get_params()
```

filters attributes based on `_` prefixes.

sklearn expects parameters to match the **constructor signature exactly**.

This difference might cause issues with cloning.

---

## main compatibility issue

The error reported in the issue is:

```id="8w2g2j"
AttributeError: 'GAM' object has no attribute '__sklearn_tags__'
```

Which suggests sklearn tries to introspect the estimator and fails.

Possible solutions might include:

* inheriting from BaseEstimator
* implementing `__sklearn_tags__`
* aligning get_params behavior

But I haven't tested the fixes yet.

---

## quick summary

sklearn estimators must follow these rules:

1. parameters defined in `__init__`
2. implement `fit()`
3. implement `predict()` (usually)
4. implement `get_params()` and `set_params()`
5. support cloning
6. provide estimator tags

pyGAM mostly follows the interface but there are some differences which might explain the compatibility issues.

---

## things I want to check later

Some things I haven't tested yet:

* running `sklearn.utils.estimator_checks`
* whether `clone(GAM())` works correctly
* whether pipelines work with GAM

These might reveal additional issues.

---

(end of notes)
