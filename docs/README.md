# Getting Started with PyTorch Lattice

DESCRIPTION HERE

---

ADD OTHER MISSING TAGS HERE e.g. Documentation, Downloads, PyPI version

[![GitHub stars](https://img.shields.io/github/stars/ControlAI/pytorch-lattice.svg)](https://github.com/ControlAI/pytorch-lattice/stargazers)
[![](https://github.com/ControlAI/pytorch-lattice/actions/workflows/test.yml/badge.svg?branch=main)](https://github.com/ControlAI/pytorch-lattice/actions/workflows/test.yml)
[![GitHub issues](https://img.shields.io/github/issues/ControlAI/pytorch-lattice.svg)](https://github.com/ControlAI/pytorch-lattice/issues)
[![GitHub license](https://img.shields.io/github/license/ControlAI/pytorch-lattice.svg)](https://github.com/ControlAI/pytorch-lattice/blob/main/LICENSE)
[![Github discussions](https://img.shields.io/github/discussions/ControlAI/pytorch-lattice)](https:github.com/ControlAI/pytorch-lattice/discussions)

---

## Installation

Install PyTorch Lattice and start training and analyzing calibrated models in minutes.

```sh
$ pip install pytorch-lattice
```

## Quickstart

!!! info "Use the [Quickstart Colab](https://colab.research.google.com/drive/1KSzTXCIXSo20w7s_3c0yKcYVTOEADm6H?usp=sharing) to get started even faster!"

### Step 1. Import the package

First, import the PyTorch Lattice library:

```py
import pytorch_lattice as pyl
```

### Step 2. Load data and fit a classifier

Load the UCI Statlog (Heart) dataset. Then create a base classifier and fit it to the data. Creating the base classifier requires only the feature names.

```py
X, y = pyl.datasets.heart()
clf = pyl.Classifier(X.columns).fit(X, y)
```

### Step 3. Plot feature calibrators

Now that you've trained a classifier, you can plot the feature calibrators to better understand how the model is understanding each feature.

```py
pyl.plot.calibrators(clf)
```

NEED TO ADD IMAGES HERE

## What's Next?

### Concepts

See link to concepts + add more description

### Walkthroughs & Examples

See link to walkthroughs for full guided walkthroughs with explanations on specific datasets

Also check out our python examples of pure code in the repo.

### API Reference

Check out the API Reference for full details on all classes, methods, functions, etc.

## License

This project is licensed under the terms of the [MIT License](https://github.com/ControlAI/pytorch-lattice/blob/main/LICENSE).
