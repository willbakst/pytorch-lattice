# Getting Started with PyTorch Lattice

A PyTorch implementation of constrained optimization and modeling techniques

---

[![GitHub stars](https://img.shields.io/github/stars/ControlAI/pytorch-lattice.svg)](https://github.com/ControlAI/pytorch-lattice/stargazers)
[![Documentation](https://img.shields.io/badge/docs-available-brightgreen)](https://broken.github.io/pytorch-lattice)
[![](https://github.com/ControlAI/pytorch-lattice/actions/workflows/test.yml/badge.svg?branch=main)](https://github.com/ControlAI/pytorch-lattice/actions/workflows/test.yml)
[![GitHub issues](https://img.shields.io/github/issues/ControlAI/pytorch-lattice.svg)](https://github.com/ControlAI/pytorch-lattice/issues)
[![Github discussions](https://img.shields.io/github/discussions/ControlAI/pytorch-lattice)](https:github.com/ControlAI/pytorch-lattice/discussions)
[![GitHub license](https://img.shields.io/github/license/ControlAI/pytorch-lattice.svg)](https://github.com/ControlAI/pytorch-lattice/blob/main/LICENSE)
[![PyPI version](https://img.shields.io/pypi/v/pytorch-lattice.svg)](https://pypi.python.org/pypi/pytorch-lattice)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/pytorch-lattice.svg)](https://pypi.python.org/pypi/pytorch-lattice)

---

## Installation

Install PyTorch Lattice and start training and analyzing calibrated models in minutes.

```sh
$ pip install pytorch-lattice
```

## Quickstart

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

### Step 3. Plot a feature calibrator

Now that you've trained a classifier, you can plot the feature calibrators to better understand how the model is understanding each feature.

```py
pyl.plots.calibrator(clf.model, "thal")
```

![Thal Calibrator](img/thal_calibrator.png)

### Step 4. What's Next?

-   Check out the [Concepts](concepts/classifier.md) section to dive deeper into the library and the core features that make it powerful, such as [calibrators](concepts/calibrators.md) and [shape constraints](concepts/shape_constraints.md).
-   You can follow along with more detailed [walkthroughs](walkthroughs/uci_adult_income.md) to get a better understanding of how to utilize the library to effectively model your data. You can also take a look at [code examples](https://github.com/ControlAI/pytorch-lattice/tree/main/examples) in the repo.
-   The [API Reference](api/layers.md) contains full details on all classes, methods, functions, etc.

## Related Research

- [Monotonic Kronecker-Factored Lattice](https://openreview.net/forum?id=0pxiMpCyBtr), William Taylor Bakst, Nobuyuki Morioka, Erez Louidor, International Conference on Learning Representations (ICLR), 2021
- [Multidimensional Shape Constraints](https://proceedings.mlr.press/v119/gupta20b.html), Maya Gupta, Erez Louidor, Oleksandr Mangylov, Nobu Morioka, Taman Narayan, Sen Zhao, Proceedings of the 37th International Conference on Machine Learning (PMLR), 2020
- [Deontological Ethics By Monotonicity Shape Constraints](https://arxiv.org/abs/2001.11990), Serena Wang, Maya Gupta, International Conference on Artificial Intelligence and Statistics (AISTATS), 2020
- [Shape Constraints for Set Functions](http://proceedings.mlr.press/v97/cotter19a.html), Andrew Cotter, Maya Gupta, H. Jiang, Erez Louidor, Jim Muller, Taman Narayan, Serena Wang, Tao Zhu. International Conference on Machine Learning (ICML), 2019
- [Diminishing Returns Shape Constraints for Interpretability and Regularization](https://papers.nips.cc/paper/7916-diminishing-returns-shape-constraints-for-interpretability-and-regularization), Maya Gupta, Dara Bahri, Andrew Cotter, Kevin Canini, Advances in Neural Information Processing Systems (NeurIPS), 2018
- [Deep Lattice Networks and Partial Monotonic Functions](https://research.google.com/pubs/pub46327.html), Seungil You, Kevin Canini, David Ding, Jan Pfeifer, Maya R. Gupta, Advances in Neural Information Processing Systems (NeurIPS), 2017
- [Fast and Flexible Monotonic Functions with Ensembles of Lattices](https://papers.nips.cc/paper/6377-fast-and-flexible-monotonic-functions-with-ensembles-of-lattices), Mahdi Milani Fard, Kevin Canini, Andrew Cotter, Jan Pfeifer, Maya Gupta, Advances in Neural Information Processing Systems (NeurIPS), 2016
- [Monotonic Calibrated Interpolated Look-Up Tables](http://jmlr.org/papers/v17/15-243.html), Maya Gupta, Andrew Cotter, Jan Pfeifer, Konstantin Voevodski, Kevin Canini, Alexander Mangylov, Wojciech Moczydlowski, Alexander van Esbroeck, Journal of Machine Learning Research (JMLR), 2016
- [Optimized Regression for Efficient Function Evaluation](http://ieeexplore.ieee.org/document/6203580/), Eric Garcia, Raman Arora, Maya R. Gupta, IEEE Transactions on Image Processing, 2012
- [Lattice Regression](https://papers.nips.cc/paper/3694-lattice-regression), Eric Garcia, Maya Gupta, Advances in Neural Information Processing Systems (NeurIPS), 2009

## Contributing

PyTorch Lattice welcomes contributions from the community! See the [contribution guide](CONTRIBUTING.md) for more information on the development workflow. For bugs and feature requests, visit our [GitHub Issues](https://github.com/ControlAI/pytorch-lattice/issues) and check out our [templates](https://github.com/ControlAI/pytorch-lattice/tree/main/.github/ISSUE_TEMPLATES).

## How To Help

Any and all help is greatly appreciated! Check out our page on [how you can help](HELP.md).

## Roadmap

Check out the our [roadmap](https://github.com/orgs/ControlAI/projects/1/views/1) to see what's planned. If there's an item that you really want that isn't assigned or in progress, take a stab at it!

## Versioning

PyTorch Lattice uses [Semantic Versioning](https://semver.org/).

## License

This project is licensed under the terms of the [MIT License](https://github.com/ControlAI/pytorch-lattice/blob/main/LICENSE).
