# pygam architecture notes (reading the codebase)

These are my notes while going through the pygam source code.
This is not meant to be perfect documentation, just trying to understand how the library is structured internally.

I mainly looked at the following modules:

* pygam.py
* distributions.py
* links.py
* penalties.py
* callbacks.py
* core.py

---

## High level idea

pyGAM implements **Generalized Additive Models (GAMs)** using spline functions.

Conceptually the model looks like:

```
y = β0 + f1(x1) + f2(x2) + ... + fk(xk)
```

where each `f` is a spline.

Instead of fitting a single linear function, the model builds **basis expansions** for each feature and learns coefficients for those.

So internally it becomes something like:

```
y ≈ B * β
```

where

* `B` = spline basis matrix
* `β` = coefficients

The model training then becomes a penalized least squares problem.

---

## Main class

The main class is in `pygam/pygam.py`.

```
class GAM(Core, MetaTermMixin):
```

This seems to be the central estimator that everything goes through.

Important attributes I noticed:

* `coef_` → learned coefficients
* `statistics_` → stores model statistics
* `logs_` → callback outputs

One thing I noticed is that `GAM` **does not inherit from sklearn BaseEstimator** which might explain why sklearn tools break later.

---

## Fit pipeline (rough)

From what I understood the training process roughly does this:

1. Validate inputs
2. Build spline terms
3. Construct model matrix
4. Run PIRLS optimization

The PIRLS algorithm is implemented in `_pirls()`.

---

## model matrix construction

There is a function called:

```
_modelmat()
```

which builds the basis matrix.

This calls into the **TermList** object which holds different feature terms.

Example terms:

* `SplineTerm`
* `LinearTerm`
* `TensorTerm`
* `FactorTerm`
* `Intercept`

Each term knows how to generate its own basis functions.

So final matrix looks like

```
[B1 | B2 | B3 | ... | Bk]
```

where each block corresponds to a feature.

---

## PIRLS optimization

The optimization loop is inside `_pirls`.

PIRLS = Penalized Iteratively Reweighted Least Squares.

From reading the code, the steps seem to be something like:

1. compute linear predictor

```
lp = B * coef
```

2. apply link function

```
mu = link.mu(lp)
```

3. compute weights matrix W

4. compute pseudo response

```
z = lp + (y - mu) * gradient
```

5. solve weighted least squares

The code uses QR decomposition + SVD for numerical stability which is interesting.

I am not 100% sure but I think this is similar to how GLMs are fitted.

---

## penalty system

Penalties are implemented in `penalties.py`.

The default penalty seems to be a **second derivative penalty**.

This penalizes curvature of the spline so that the functions remain smooth.

Example penalty functions:

* `derivative`
* `l2`
* `periodic`
* `none`

There are also constraint penalties like:

* monotonic increasing
* monotonic decreasing
* convex
* concave

These are added to the optimization matrix.

---

## distributions

The distribution classes define the likelihood.

Examples:

```
NormalDist
PoissonDist
BinomialDist
GammaDist
InvGaussDist
```

Each distribution implements:

* `log_pdf`
* `deviance`
* `V(mu)` (variance function)

This is very similar to GLM design.

---

## link functions

Link functions convert between the **mean of the distribution** and the **linear predictor**.

Examples implemented:

* IdentityLink
* LogLink
* LogitLink
* InverseLink
* InvSquaredLink

Typical example:

```
lp = log(mu)
mu = exp(lp)
```

---

## callbacks

Callbacks run during optimization.

Examples:

* Deviance
* Accuracy
* Diffs
* Coef

These seem to be used mainly for logging.

Inside the PIRLS loop I saw something like:

```
_on_loop_start()
_on_loop_end()
```

which triggers callbacks.

---

## internal utilities

There are also a lot of helper utilities in `utils.py`.

Things like:

* array validation
* spline basis construction
* matrix operations
* Cholesky factorization

I didn't go through all of them yet because the file is quite large.

---

## important observation (sklearn compatibility)

While reading the code I noticed something interesting.

The `GAM` class does **not inherit from sklearn BaseEstimator**.

Also there is no implementation of:

```
__sklearn_tags__()
```

In sklearn >=1.7 this seems to be required for estimator introspection.

This might be the reason why `RandomizedSearchCV` crashes with pygam.

---

## very rough architecture diagram

Not completely sure if this is 100% accurate but roughly:

```
        GAM
         |
    -----------
    |    |    |
 Terms Dist  Link
    |
 TermList
    |
SplineTerm etc
```

Then during fitting:

```
X -> Terms -> Basis Matrix -> PIRLS -> coefficients
```

---

## things I still don't fully understand

Some parts of the code are still a bit confusing:

* how exactly `TermList.compile()` works
* how tensor splines are implemented
* how edof is calculated
* why SVD is used instead of just solving the normal equations

Will probably need to go through those parts again.

---

## conclusion

After going through the code it seems that:

* pyGAM is basically a **GLM with spline basis expansions**
* optimization is done using **PIRLS**
* the library has modular components for distributions and links
* sklearn compatibility might break because the estimator does not follow the sklearn API exactly

These notes are just from a first pass through the codebase so there might be some misunderstandings.
