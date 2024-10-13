import tensorflow as tf
from tensorflow import keras as keras
import numpy as np


class MultiplicativeAttention(keras.layers.Layer):
    def __init__(self, units):
        super().__init__()

        # dense layer to compute multiplicative style scores
        self.score_dense = keras.layers.Dense(units, use_bias=False)

    def call(self, inputs: tuple[tf.Tensor, tf.Tensor]) -> tuple[tf.Tensor, tf.Tensor]:
        query, key = (
            inputs  # query are the hidden states of decoder, and key are the hidden states of encoder
        )

        # Compute attention scores QxWxK^T
        scores = tf.matmul(query, self.score_dense(key), transpose_b=True)

        # Apply softmax to get attention weights
        attention_weights = tf.nn.softmax(scores, axis=-1)

        # Compute context vector as weighted average of encoder's hidden states
        context = tf.matmul(attention_weights, key)

        return context, attention_weights


class Seq2SeqAttention:
    def __init__(self, latent_dim: int, dim_vectors: int, n_steps: int):
        self.dim_vectors = dim_vectors
        self.n_steps = n_steps

        # encoder
        encoder_inputs = keras.layers.Input(shape=(None, dim_vectors))
        encoder_o, encoder_h, encoder_c = keras.layers.LSTM(
            units=latent_dim, return_sequences=True, return_state=True
        )(encoder_inputs)
        encoder_states = [encoder_h, encoder_c]

        # decoder
        decoder_inputs = keras.layers.Input(shape=(None, dim_vectors))
        decoder_lstm = keras.layers.LSTM(
            units=latent_dim, return_sequences=True, return_state=True
        )
        decoder_o, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)

        attention_layer = MultiplicativeAttention(latent_dim)
        context, attention_weights = attention_layer([decoder_o, encoder_o])

        # Concatenate attention output and decoder LSTM output
        decoder_concat_input = keras.layers.Concatenate(axis=-1)([decoder_o, context])

        # last dense layer
        decoder_dense = keras.layers.Dense(dim_vectors)
        decoder_outputs = decoder_dense(decoder_concat_input)

        # Define the full model that will be trained with teacher forcing
        self.model = keras.models.Model(
            [encoder_inputs, decoder_inputs], decoder_outputs
        )

        # encoder for inference
        self.encoder_model = keras.models.Model(
            encoder_inputs, [encoder_o] + encoder_states
        )

        # adapted decoder for inference, using one output decoder as the next input of it
        decoder_state_input_h = keras.layers.Input(shape=(latent_dim,))
        decoder_state_input_c = keras.layers.Input(shape=(latent_dim,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

        encoder_o_inf = keras.layers.Input(shape=(None, latent_dim))

        decoder_inf_o, state_h, state_c = decoder_lstm(
            decoder_inputs, initial_state=decoder_states_inputs
        )
        context_inf, attention_weights_inf = attention_layer(
            [decoder_inf_o, encoder_o_inf]
        )
        decoder_inf_concat = keras.layers.Concatenate(axis=-1)(
            [decoder_inf_o, context_inf]
        )
        decoder_inf_outputs = decoder_dense(decoder_inf_concat)

        self.decoder_model = keras.models.Model(
            [decoder_inputs, encoder_o_inf] + decoder_states_inputs,
            [decoder_inf_outputs, state_h, state_c],
        )

    def compile(
        self,
        optimizer: keras.optimizers.Optimizer,
        loss: str,
        metrics: list[keras.metrics.Metric],
    ):
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def fit(
        self,
        X_train: np.ndarray,
        decoder_input_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        decoder_input_val: np.ndarray,
        y_val: np.ndarray,
        **kwargs
    ):

        self.model.fit(
            [X_train, decoder_input_train],
            y_train,
            validation_data=([X_val, decoder_input_val], y_val),
            **kwargs
        )

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        encoder_o, state_h, state_c = self.encoder_model.predict(X_test)
        last_values = X_test[:, -1:, :]
        pred_test = np.zeros(shape=(X_test.shape[0], self.n_steps, self.dim_vectors))

        for i in range(self.n_steps):
            last_values, state_h, state_c = self.decoder_model.predict(
                [last_values, encoder_o, state_h, state_c]
            )
            pred_test[:, i, :] = last_values[:, 0, :]

        return pred_test
