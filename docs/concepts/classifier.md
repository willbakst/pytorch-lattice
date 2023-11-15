# Classifier

The [`Classifier`](/pytorch-lattice/api/classifier) class is a high-level wrapper around the calibrated modeling functionality to make it extremely easy to fit a calibrated model to a classification task. The class uses declarative configuration and automatically handles the data preparation, feature configuration, model creation, and model training necessary for properly training a calibrated model.

## Initialization

The only required parameter for creating a classifier is the list of features to use:

```py
clf = pyl.Classifier(["list", "of", "features"])
```

You do not need to include all of the feature present in your dataset. When you specify only a subset of the features, the classifier will automatically handle selecting only those features for training.

## Fitting

Fitting the classifier to your data is as simple as calling `fit(...)`:

```py
clf.fit(X, y)
```

You can additionally further specify hyperparameters used for fitting such as `epochs`, `batch_size`, and `learning_rate`. Just pass the values in as parameters:

```py
clf.fit(X, y, epochs=100, batch_size=512, learning_rate=1e-4)
```

When you call fit, the classifier will train a new model, overwriting any previously trained model. If you want to run a hyperparameter optimization job to find the best setting of hyperparameters, you can first extract the trained model before calling fit again:

```py
models = []
for epochs, batch_size, learning_rate in hyperparameters:
    clf.fit(X, y, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate)
    models.append(clf.model)
```

The benefit of extracting the model is that you can reuse the same classifier configuration; however, you can also always create a new classifier for each setting instead:

```py
clfs = []
for epochs, batch_size, learning_rate in hyperparameters:
    clf = pyl.Classifier(X.columns).fit(
        X, y, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate
    )
    clfs.append(clf)
```

## Generate Predictions

You can generate predictions using the `predict(...)` function:

```py
probabilities = clf.predict(X)
logits = clf.predict(X, logits=True)
```

Just make sure that the input `pd.DataFrame` contains all of the features the classifier is expecting.

## Model Configuration

To configure the type of calibrated model to use for the classifier, you can additionally provide a model configuration during initialization:

```py
model_config = pyl.model_configs.LinearConfig(use_bias=False)
clf = pyl.Classifier(["list", "of", "features"], model_config)
```

See [Model Types](model_types.md) for more information on the supported model types and [model_configs](/pytorch-lattice/api/model_configs) for more information on configuring these models in a classifier. 

## Feature Configuration

When you first initialize a calibrator, all features will be initialized using default values. You can further specify configuration options for features by retrieve the feature's configuration from the classifier and calling the corresponding function to set that option:

```py
clf.configure("feature").monotonicity("increasing").num_keypoints(10)
```

See [feature_configs](/pytorch-lattice/api/feature_config/) for all of the available configuration options.

## Categorical Features

If the value type for a feature in the dataset is not numerical (e.g. string), the classifier will automatically handle the feature as categorical, using all unique categories present in the dataset as the categories for the calibrator.

If you want the classifier to handle a discrete numerical value as a categorical feature, simply convert the values to strings:

```py
X["categorical_feature"] = X["categorical_feature"].astype(str)
```

Additionally you can specify a list of categories to use as a configuration option:

```py
clf.configure("categorical_feature").categories(["list", "of", "categories"])
```

Any category in the dataset that is not present in the configured category list will be lumped together into a missing category bucket, which will also have a learned calibration. This can be particularly useful if there are categories in your dataset that appear in very few examples.

## Saving & Loading

The `Classifier` class also provides easy save/load functionality so that you can save your classifiers and load them as necessary to generate predictions:

```py
clf.save("path/to/dir")
loaded_clf = pyl.Classifier.load("path/to/dir")
```
