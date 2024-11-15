from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
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

def create_lagged_features_with_dynamics(
    df: pd.DataFrame, columns: list[str], max_lag: int, window_size: Optional[int] = None, dropna: bool = True
):
    """
    Generate lagged predictors along with windowed (or global) average rate of change 
    and acceleration for specified columns in a pandas DataFrame.

    Parameters:
    - df: pandas DataFrame containing the original data.
    - columns: list of column names for which to generate lagged predictors.
    - max_lag: maximum number of lags to generate.
    - window_size: size of the windows for computing dynamics. If None, use the entire lagged period.
    - dropna: boolean indicating whether to drop rows with NaN values resulting from lagging.

    Returns:
    - A pandas DataFrame with original, lagged predictors, windowed/global average rate of change,
      and windowed/global average acceleration.
    """
    # Start by creating lagged features
    result_df = create_lagged_features(df, columns, max_lag, dropna=False)

    for col in columns:
        lagged_cols = [f"{col}_lag_{lag}" for lag in range(1, max_lag + 1)]
        
        # Handle the special case for no windows
        if window_size is None or window_size >= max_lag:
            # Compute first differences across all lags
            first_diffs = result_df[lagged_cols].sub(result_df[lagged_cols].shift(-1, axis=1), axis=1)
            result_df[f"{col}_avg_rate_of_change"] = first_diffs.mean(axis=1)
            
            # Compute second differences across all lags
            second_diffs = first_diffs.sub(first_diffs.shift(-1, axis=1), axis=1)
            result_df[f"{col}_avg_acceleration"] = second_diffs.mean(axis=1)
        else:
            # Divide the lags into windows
            num_windows = max_lag // window_size
            for w in range(num_windows):
                # Get the columns for this window
                window_cols = lagged_cols[w * window_size: (w + 1) * window_size]
                
                # Compute first differences
                first_diffs = result_df[window_cols].sub(result_df[window_cols].shift(-1, axis=1), axis=1)
                result_df[f"{col}_window_{w+1}_avg_rate_of_change"] = first_diffs.mean(axis=1)
                
                # Compute second differences
                second_diffs = first_diffs.sub(first_diffs.shift(-1, axis=1), axis=1)
                result_df[f"{col}_window_{w+1}_avg_acceleration"] = second_diffs.mean(axis=1)

    # Drop rows with NaN values if specified
    if dropna:
        result_df = result_df.dropna().reset_index(drop=True)
    
    return result_df



def create_lagged_difference_features(
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
    - A pandas DataFrame with original and lagged difference predictors.
    """
    result_df = df.copy()
    for col in columns:
        for lag in range(1, max_lag + 1):
            result_df[f"{col}_lag_diff_{lag}"] = df[col].shift(lag-1) - df[col].shift(lag)

    if dropna:
        result_df = result_df.dropna().reset_index(drop=True)

    return result_df

def process_inputs_all(input_dfs: dict[str, pd.DataFrame], target: str | list[str], add_location_intercept: bool = False):
    # Combine inputs into single sorted data frame
    input_df = pd.concat([df for _, df in input_dfs.items()])
    if add_location_intercept:
        dummies = pd.get_dummies(input_df["location"], drop_first=False)
        input_df = pd.concat([input_df, dummies], axis=1)

    X = input_df.sort_values(["date"]).reset_index(drop=True)

    # Extract dates and location
    dates, locations = X["date"], X["location"]
    if not isinstance(target, list):
        target = [target]

    y = X[target]  # Target
    X = X.drop(target + ["date", "location"], axis=1)  # Features

    return dates, locations, X, y


def withhold_test_locations_and_split(
    X: pd.DataFrame,
    y: pd.DataFrame,
    locations: pd.Series,
    withheld_locations: list[str],
    test_size=0.2,
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


class SelectivePressureData(data.Dataset):
    def __init__(self, X, y):
        super().__init__()
        self.x = X
        self.y = y

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

class SelectivePressureTSData(data.Dataset):
    def __init__(self, X, y, locations, dates, sequence_length):
        self.X = X
        self.y = y
        self.locations_vec = locations
        self.dates_vec = dates
        self.sequence_length = sequence_length
        
        # Extract unique locations
        self.locations = locations.unique()
        
        # Store time-series data per location
        self.data = {
            location: (self.X[self.locations_vec == location], self.y[self.locations_vec == location])
            for location in self.locations
        }
        
    def __len__(self):
        return len(self.locations)

    def __getitem__(self, idx):
        # Get data for the selected location
        location = self.locations[idx]
        x,y = self.data[location]
        
        # Randomly select the starting point of the sequence
        max_start_idx = len(x) - self.sequence_length
        if max_start_idx <= 0:
            raise ValueError(f"Sequence length {self.sequence_length} is too long for location {location}.")
        
        start_idx = np.random.randint(0, max_start_idx + 1)
        end_idx = start_idx + self.sequence_length
        return x[start_idx:end_idx], y[start_idx:end_idx]


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
            if num_batches * batch_size < _y.shape[0]:
                batches.append(
                    (_X[num_batches * batch_size :], _y[num_batches * batch_size :])
                )
    return batches
