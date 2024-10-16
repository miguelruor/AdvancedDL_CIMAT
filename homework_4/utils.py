from abc import ABC, abstractmethod
from tensorflow import keras as keras
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from models import TimeSeriesModel
from typing import Callable


def create_dataset(
    timeseries: np.ndarray, T: int, m: int, indices: list[int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Split the given timeseries (shape (N, n_features)) into input sequence of length T, target sequence of length m
    # and decoder input sequence of length m (target sequence shifted back by one time step).
    # Sequences are made according to the given indices.
    # Returns the input sequence, target sequence and decoder input sequnce

    X = np.array([timeseries[i : i + T] for i in indices])
    X, y, decoder_input = (
        X[:, : T - m, :].copy(),
        X[:, -m:, :].copy(),
        X[:, -m - 1 : -1, :].copy(),
    )

    return X, y, decoder_input


# Abstract class for typing util functions
class CustomModel(ABC):
    def __init__(self):
        self.model: keras.Model = None

    @abstractmethod
    def fit(
        self,
        X_train: np.ndarray,
        decoder_input_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        decoder_input_val: np.ndarray,
        y_val: np.ndarray,
        **kwargs,
    ):
        pass

    @abstractmethod
    def predict(self, X_test: np.ndarray):
        pass

    @abstractmethod
    def compile(
        self,
        optimizer: keras.optimizers.Optimizer,
        loss: str,
        metrics: list[keras.metrics.Metric],
    ):
        pass


def plot_training_metrics(
    model: keras.Model,
    metric: str,
    metric_name: str,
    loss_name: str,
    n_epochs: int,
    figsize: tuple[int, int] = (14, 5),
    plot_all_loss: bool = False,
):
    # function to plot loss function and metric function when training the given CustomModel

    start_loss = 0 if plot_all_loss else 1

    plt.figure(figsize=figsize)

    plt.subplot(1, 2, 1)
    plt.plot(
        range(start_loss, n_epochs),
        model.history.history["loss"][1:],
        label="Entrenamiento",
    )
    plt.plot(
        range(start_loss, n_epochs),
        model.history.history["val_loss"][1:],
        label="Validación",
    )
    plt.title("Función de pérdida")
    plt.xlabel("Época")
    plt.ylabel(loss_name)

    plt.subplot(1, 2, 2)
    plt.plot(model.history.history[metric], label="Entrenamiento")
    plt.plot(model.history.history[f"val_{metric}"], label="Validación")
    plt.title(metric_name)
    plt.xlabel("Época")
    plt.ylabel(metric)

    plt.legend()
    plt.show()


def train_model_and_plot(
    model: keras.Model,
    epochs: int,
    adam_optimizer_params: dict,
    X: np.ndarray,
    decoder_input: np.ndarray,
    y: np.ndarray,
    X_val: np.ndarray,
    decoder_input_val: np.ndarray,
    y_val: np.ndarray,
    plot_all_loss: bool = False,
):
    # function to train given CustomModel and plot loss function and metric function

    model.compile(
        optimizer=keras.optimizers.Adam(**adam_optimizer_params),
        loss="mse",
        metrics=[keras.metrics.MeanAbsoluteError()],
    )

    model.fit(
        [X, decoder_input],
        y,
        validation_data=([X_val, decoder_input_val], y_val),
        batch_size=64,
        epochs=epochs,
    )

    plot_training_metrics(
        model=model,
        metric="mean_absolute_error",
        metric_name="Error absoluto medio",
        loss_name="Error cuadrático medio",
        n_epochs=epochs,
        plot_all_loss=plot_all_loss,
    )


def predictions_analysis(
    model: TimeSeriesModel,
    X_test: np.ndarray,
    y_test: np.ndarray,
    y_test_raw: np.ndarray,
    features_name: list[str],
    back2rawdata: Callable[[np.ndarray], np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Function to predict from given data and given TimeSeriesModel, and compute mean absolute error from in preprocessed data scale and real scale

    data_dim = X_test.shape[-1]

    pred = model.predict(X_test)  # predictions
    pred_flat = pred.reshape(-1, data_dim)
    y_test_flat = y_test.reshape(-1, data_dim)  # ground truth

    mae = mean_absolute_error(y_test_flat, pred_flat, multioutput="raw_values")

    print("\nErrores absolutos medios en la escala normalizada:")
    for i, feature in enumerate(features_name):
        print(feature, ":", mae[i])

    pred_raw = back2rawdata(pred)  # predicted prices
    pred_raw_flat = pred_raw.reshape(-1, data_dim)
    y_test_raw_flat = y_test_raw.reshape(-1, data_dim)

    mae_raw = mean_absolute_error(
        y_test_raw_flat, pred_raw_flat, multioutput="raw_values"
    )

    print("\nErrores absolutos medios en la escala de los precios:")
    for i, feature in enumerate(features_name):
        print(feature, ":", mae_raw[i])

    return pred, pred_raw, mae, mae_raw


def plot_predictions(
    predictions: np.ndarray,
    dataset_df: pd.DataFrame,
    rand_ind: np.ndarray,
    rand_times: list[pd.Timestamp],
    T: int,
    m: int,
    window_size: int = 40,
):
    # function to plot predictions in random times (rand_times) and for each financial asset.
    # rand_ind are the indices that correspond to rand_times in the predictions array
    # columns in plot correspond to the selected times, and rows correspond to financial assets.

    assets = dataset_df.columns

    n_rows, n_cols = 7, len(rand_ind)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(30, 45))

    fig.suptitle("Predicciones de precios", fontsize=22)

    # Iterate over each subplot (i, j) to plot prices of i-th financial asset and visually evaluate performance of prediction j
    for j in range(n_cols):
        jth_time = rand_times[
            j
        ]  # time t corresponding to sequence (x_t, x_{t+1}, ..., x_{t+\tau-1})
        jth_data = dataset_df.loc[
            jth_time
            + pd.Timedelta(hours=T - window_size) : jth_time
            + pd.Timedelta(hours=T - 1)
        ]  # from the window of size T starting at t, just consider the last window_size points

        pred_times = jth_data.index[-m:]  # predicted times
        last_time = jth_data.index[-m - 1]  # last time t+\tau -1

        for i in range(n_rows):
            ith_asset = assets[i]
            last_price = dataset_df.loc[last_time][ith_asset]

            axes[i, j].plot(jth_data[ith_asset], label="Precios")
            axes[i, j].plot(
                [last_time] + [t for t in pred_times],
                [last_price] + [pt for pt in predictions[rand_ind[j], :, i]],
                color="r",
                alpha=0.3,
            )
            axes[i, j].axvline(
                last_time,
                color="orange",
                linestyle="--",
                label="Último tiempo antes de la predicción",
            )
            axes[i, j].scatter(
                [last_time] + [t for t in pred_times],
                [last_price] * (m + 1),
                color="b",
                label="Baseline",
            )
            axes[i, j].scatter(
                pred_times,
                predictions[rand_ind[j], :, i],
                color="r",
                label="Predicciones",
            )
            axes[i, j].set_title(
                f"{ith_asset}\nÚltimo tiempo antes de la predicción: {last_time}",
                fontsize=12,
            )
            axes[i, j].xaxis.set_major_locator(plt.MaxNLocator(7))
            axes[i, j].tick_params(axis="x", labelrotation=45)
            axes[i, j].legend(fontsize=12)

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.show()


def predictions_analysis(
    model: TimeSeriesModel,
    X_test: np.ndarray,
    y_test: np.ndarray,
    y_test_raw: np.ndarray,
    back2rawdata: Callable[[np.ndarray], np.ndarray],
    dataset_df: pd.DataFrame,
    T: int,
    m: int,
    rand_ind: np.ndarray,
    rand_times: list[pd.Timestamp],
    window_size: int = 40,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Function to predict from given data and given TimeSeriesModel, and compute mean absolute error from in preprocessed data scale and real scale
    assets = dataset_df.columns
    data_dim = X_test.shape[-1]

    pred = model.predict(X_test)  # predictions
    pred_flat = pred.reshape(-1, data_dim)
    y_test_flat = y_test.reshape(-1, data_dim)  # ground truth

    mae = mean_absolute_error(y_test_flat, pred_flat, multioutput="raw_values")

    print("\nErrores absolutos medios en la escala normalizada:")
    for i, asset in enumerate(assets):
        print(asset, ":", mae[i])

    pred_raw = back2rawdata(pred)  # predicted prices
    pred_raw_flat = pred_raw.reshape(-1, data_dim)
    y_test_raw_flat = y_test_raw.reshape(-1, data_dim)

    mae_raw = mean_absolute_error(
        y_test_raw_flat, pred_raw_flat, multioutput="raw_values"
    )

    print("\nErrores absolutos medios en la escala de los precios:")
    for i, asset in enumerate(assets):
        print(asset, ":", mae_raw[i])

    plot_predictions(
        predictions=pred_raw,
        dataset_df=dataset_df,
        rand_ind=rand_ind,
        rand_times=rand_times,
        T=T,
        m=m,
        window_size=window_size,
    )

    return pred, pred_raw, mae, mae_raw
