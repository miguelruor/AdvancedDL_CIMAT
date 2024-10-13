from abc import ABC, abstractmethod
from tensorflow import keras as keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


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
    model: CustomModel,
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
        model.model.history.history["loss"][1:],
        label="Entrenamiento",
    )
    plt.plot(
        range(1, n_epochs),
        model.model.history.history["val_loss"][1:],
        label="Validación",
    )
    plt.title("Función de pérdida")
    plt.xlabel("Época")
    plt.ylabel(loss_name)

    plt.subplot(1, 2, 2)
    plt.plot(model.model.history.history[metric], label="Entrenamiento")
    plt.plot(model.model.history.history[f"val_{metric}"], label="Validación")
    plt.title(metric_name)
    plt.xlabel("Época")
    plt.ylabel(metric)

    plt.legend()
    plt.show()


def train_model_and_plot(
    model: CustomModel,
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
        X,
        decoder_input,
        y,
        X_val,
        decoder_input_val,
        y_val,
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


class LocalNormalization:
    def __init__(self, sequences: np.ndarray):
        # compute vector of means and standard deviations, for each sequence

        self.means = sequences.mean(
            axis=1
        )  # vector mu of means for each sequence, i.e., mu[i] is the vector of means for each feature of sequence i
        self.stds = sequences.std(
            axis=1
        )  # vector of standard deviations for each sequence

    def transform(self, sequences: np.ndarray) -> np.ndarray:
        sequences_norm = np.zeros(sequences.shape)

        for i in range(sequences.shape[0]):
            sequences_norm[i] = (sequences[i] - self.means[i]) / self.stds[i]

        return sequences_norm

    def inverse_transform(self, sequences_norm: np.ndarray) -> np.ndarray:
        sequences_raw = np.zeros(sequences_norm.shape)

        for i in range(sequences_norm.shape[0]):
            sequences_raw[i] = self.stds[i] * sequences_norm[i] + self.means[i]

        return sequences_raw


class LocalMinMaxScaling:
    def __init__(self, sequences: np.ndarray):
        # compute vector of minimums and maximums, for each sequence

        self.min = sequences.min(
            axis=1
        )  # vector of minimums for each sequence, i.e.,  min[i] is the vector of minimums of each feature in sequence i
        self.max = sequences.max(axis=1)  # vector of maximums for each sequence

    def transform(self, sequences: np.ndarray) -> np.ndarray:
        sequences_scaled = np.zeros(sequences.shape)

        for i in range(sequences.shape[0]):
            sequences_scaled[i] = (sequences[i] - self.min[i]) / (
                self.max[i] - self.min[i]
            )

        return sequences_scaled

    def inverse_transform(self, sequences_scaled: np.ndarray) -> np.ndarray:
        sequences_raw = np.zeros(sequences_scaled.shape)

        for i in range(sequences_scaled.shape[0]):
            sequences_raw[i] = (self.max[i] - self.min[i]) * sequences_scaled[
                i
            ] + self.min[i]

        return sequences_raw
