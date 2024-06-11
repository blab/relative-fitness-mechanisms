from functools import partial

import jax
import jax.numpy as jnp
import pandas as pd
from flax import linen as nn
from sklearn.model_selection import train_test_split


def create_lagged_features(
    df: pd.DataFrame, columns: list[str], max_lag: int, dropna: bool = True
):
    """
    Generate lagged predictors for specified columns in a pandas DataFrame.

    Parameters:
    - df: pandas DataFrame containing the original data.
    - columns: list of column names for which to generate lagged predictors.
    - max_lag: maximum number of lags to generate.
    - dropna: boolean indicating whether to drop rows with NaN values resulting from lagging.

    Returns:
    - A pandas DataFrame with original and lagged predictors.
    """
    result_df = df.copy()
    for col in columns:
        for lag in range(1, max_lag + 1):
            result_df[f"{col}_lag_{lag}"] = df[col].shift(lag)

    if dropna:
        result_df = result_df.dropna().reset_index(drop=True)

    return result_df


def process_inputs_all(input_dfs: dict[str, pd.DataFrame], target: str):
    # Combine inputs into single sorted data frame
    input_df = pd.concat([df for _, df in input_dfs.items()])
    X = input_df.sort_values(["date"]).reset_index(drop=True)

    # Extract dates and location
    dates, locations = X["date"], X["location"]

    y = X[target]  # Target
    X = X.drop([target, "date", "location"], axis=1)  # Features

    return dates, locations, X, y


def withhold_test_locations_and_split(
    X: pd.DataFrame,
    y: pd.DataFrame,
    locations: pd.Series,
    withheld_locations: list[str],
    test_size = 0.2
):

    # Withold certain locations for testing
    _X, _y, X_withheld, y_withheld = (
        X[~locations.isin(withheld_locations)],
        y[~locations.isin(withheld_locations)],
        X[locations.isin(withheld_locations)],
        y[locations.isin(withheld_locations)],
    )

    # Split to hold final section for testing best model after CV
    X_train, X_test, y_train, y_test = train_test_split(
        _X, _y, test_size=test_size, shuffle=False
    )

    X_test = pd.concat([X_withheld, X_test])
    y_test = pd.concat([y_withheld, y_test])
    return X_train, y_train, X_test, y_test


class SimpleTransformer(nn.Module):
    num_heads: int
    d_model: int  # Size of the attention representations
    output_dim: int  # Dimensionality of the regression output

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=32)(x)
        x = nn.relu(x)

        attn = nn.MultiHeadDotProductAttention(
            num_heads=4, qkv_features=16, use_bias=True
        )(x)
        x = nn.LayerNorm()(x + attn)

        x = nn.Dense(features=16)(x)
        x = nn.relu(x)

        x = nn.Dense(features=self.output_dim)(x)

        return jnp.squeeze(x, axis=-1)


def smoothness_loss(params, model, x):
    # Predict the model output
    predictions = model(params, x)

    # Shifted predictions for t+1 and t-1
    predictions_shifted_right = jnp.roll(predictions, -1, axis=0)
    predictions_shifted_left = jnp.roll(predictions, 1, axis=0)

    # Compute the finite differences second derivative
    second_derivative = (
        predictions_shifted_left - 2 * predictions + predictions_shifted_right
    )

    # Penalize large derivatives (i.e., encourage smoothness)
    return jnp.mean(jnp.square(second_derivative))


def mae_loss(params, model, x, y):
    pred = model(params, x)
    return jnp.mean(jnp.abs(pred - y))


def loss_fn(params, state, x, y, alpha):
    loss_accuracy = mae_loss(params, state.apply_fn, x, y)
    loss_smooth = smoothness_loss(params, state.apply_fn, x)
    return loss_accuracy + alpha * loss_smooth


@partial(jax.jit, static_argnums=3)
def train_step(state, x_batch, y_batch, loss_fn):
    grad_fn = jax.value_and_grad(lambda p: loss_fn(p, state, x_batch, y_batch))
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss


def create_training_batches(X, y, locations_vec, batch_size):
    batches = []
    for loc in locations_vec.unique():  # Batch by location
        idx = (loc == locations_vec)[X.index]
        if (idx).any():
            _X, _y = X.values[idx], y.values[idx]
            num_batches = _y.shape[0] // batch_size
            for i in range(num_batches):
                batches.append(
                    (
                        _X[i * batch_size : (i + 1) * batch_size],
                        _y[i * batch_size : (i + 1) * batch_size],
                    )
                )
            batches.append(
                (_X[num_batches * batch_size :], _y[num_batches * batch_size :])
            )
    return batches
