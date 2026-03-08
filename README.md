# pygam sklearn compatibility investigation

Small repo while looking into this issue:

https://github.com/dswah/pyGAM/issues/422

scikit-learn >=1.7 introduces a new estimator tag system (`__sklearn_tags__`).
Current pygam versions don't seem to implement it, which causes failures when using sklearn tools like `RandomizedSearchCV`.

I wanted to understand:

* how pygam's estimator is structured
* why sklearn introspection fails
* what minimal fixes might look like

This repo is mostly just notes + small experiments.

## repo structure

```
investigation/
    notes while reading pygam code

experiments/
    small scripts reproducing the error

patches/
    quick experiments trying possible fixes

notebooks/
    quick interactive tests
```

## reproduction

```
python experiments/failing_example.py
```

This should raise something like:

```
AttributeError: 'GAM' object has no attribute '__sklearn_tags__'
```

when using sklearn >=1.7.

## idea for fix

Two things that might help:

* implementing `__sklearn_tags__`
* inheriting from `sklearn.base.BaseEstimator`

These are tested in the `patches/` directory.

## note

This is mostly exploratory and not meant as a production patch.
Just trying to understand the compatibility issue.
