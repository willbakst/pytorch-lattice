import pytorch_lattice as pyl

# Load Data
X, y = pyl.datasets.heart()

# Fit Classifier
clf = pyl.Classifier(X.columns).fit(X, y)

# Plot Calibrator For Feature "thal"
pyl.plots.calibrator(clf.model, "thal")

# Plot Linear Coefficients For Calibrated Linear Model
pyl.plots.linear_coefficients(clf.model)
