from abc import ABC, abstractmethod
from tensorflow import keras as keras
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
):
    # function to plot loss function and metric function when training the given CustomModel

    plt.figure(figsize=figsize)

    plt.subplot(1, 2, 1)
    plt.plot(
        range(1, n_epochs),
        model.history.history["loss"][1:],
        label="Entrenamiento",
    )
    plt.plot(
        range(1, n_epochs),
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
