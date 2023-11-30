import numpy as np
import torch
from sklearn import metrics

import pytorch_lattice as pyl
from pytorch_lattice.models.features import CategoricalFeature, NumericalFeature

# Load Data
X, y = pyl.datasets.heart()

# Configure Features
features = [
    NumericalFeature("age", data=np.array(X["age"].values), monotonicity="increasing"),
    NumericalFeature(
        "trestbps", data=np.array(X["trestbps"].values), monotonicity="increasing"
    ),
    NumericalFeature(
        "chol", data=np.array(X["chol"].values), monotonicity="increasing"
    ),
    CategoricalFeature("ca", categories=X["ca"].unique().tolist()),
    CategoricalFeature(
        "thal",
        categories=["fixed", "normal", "reversible"],
        monotonicity_pairs=[("normal", "fixed"), ("normal", "reversible")],
    ),
]

# Create Model (you can replace this with CalibratedLattice to train a lattice model)
model = pyl.models.CalibratedLinear(features)

# Fit Model
optimizer = torch.optim.Adam(model.parameters(recurse=True), lr=1e-3)
loss_fn = torch.nn.BCEWithLogitsLoss()
dataset = pyl.utils.data.Dataset(X, y, features)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
for epoch in range(100):
    for inputs, labels in dataloader:
        optimizer.zero_grad()
        loss_fn(model(inputs), labels).backward()
        optimizer.step()
        model.apply_constraints()

# Generate Predictions
model.eval()
X_copy = X[["age", "trestbps", "chol", "ca", "thal"]].copy()
pyl.utils.data.prepare_features(X_copy, features)
X_tensor = torch.tensor(X_copy.values).double()
with torch.no_grad():
    preds = model(X_tensor).numpy()

# Calculate AUC
fpr, tpr, _ = metrics.roc_curve(y, preds)
print(f"Train AUC: {metrics.auc(fpr, tpr)}")
