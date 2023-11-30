from sklearn import metrics

import pytorch_lattice as pyl

# Load Data
X, y = pyl.datasets.heart()

# Fit Classifier
clf = pyl.Classifier(X.columns).fit(X, y)

# Generate Predictions
preds = clf.predict(X)

# Calculate AUC
fpr, tpr, _ = metrics.roc_curve(y, preds)
print(f"Train AUC: {metrics.auc(fpr, tpr)}")
