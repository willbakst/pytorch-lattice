from sklearn import metrics

import pytorch_lattice as pyl

# Load Data
X, y = pyl.datasets.adult()

# Create Classifier On Subset Of Features
clf = pyl.Classifier(
    [
        "age",
        "workclass",
        "education_num",
        "capital_gain",
        "capital_loss",
        "hours_per_week",
    ]
)

# Configure Feature Monotonicity
clf.configure("education_num").monotonicity("increasing")
clf.configure("capital_gain").monotonicity("increasing")

# Fit Classifier
clf.fit(X, y)

# Generate Predictions
preds = clf.predict(X)

# Calculate AUC
fpr, tpr, _ = metrics.roc_curve(y, preds)
print(f"Train AUC: {metrics.auc(fpr, tpr)}")
