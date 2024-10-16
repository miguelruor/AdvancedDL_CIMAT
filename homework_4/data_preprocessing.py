import numpy as np


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
