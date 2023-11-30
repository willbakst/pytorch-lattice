# Shape Constraints

Shape constraints play a crucial role in making calibrated models interpretable by allowing users to impose specific behavioral rules on their machine learning models. These constraints help to reduce – or even eliminate – the impact of noise and inherent biases contained in the data.

[`Monotonicity`](../api/enums.md#pytorch_lattice.enums.Monotonicity) constraints ensure that the relationship between an input feature and the output prediction consistently increases or decreases. Let's consider our house price prediction task once more. A monotonic constraint on the square footage feature would guarantee that increasing the size of the property increases the predicted price. This makes sense.

Unimodality constraints (coming soon) create a single peak in the model's output, ensuring that there is only one optimal value for a given input feature. For example, a feature for price used when predicting sales volume may be unimodal since lower prices generally lead to higher sales, but prices that are too low may indicate low quality with one single optimal price.

Convexity/Concavity constraints (coming soon) ensure that the given feature's value has a convex/concave relationship with the model's output. Looking again at the feature for price for predicting sales volume, it may be that there is a range of optimal prices and not one single optimal price, which would instead be a concavity constraint.

Trust constraints (coming soon) define the relative importance of input features depending on other features. For instance, a trust constraint can ensure that a model predicting product sales relies more on the star rating (1-5) when the number of reviews is higher, which forces the model's predictions to better align with real-world expectations and rules.

Dominance constraints (coming soon) are intended to embed that a dominant feature is more important than a weak feature. For example, you might want to constrain a model predicting click-through-rate (CTR) for a specific web link to be more sensitive to past CTR for that web link than the average CTR for the whole website.

Together, these shape constraints help create machine learning models that are both interpretable and trustworthy.
