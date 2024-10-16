import tensorflow as tf
from tensorflow import keras as keras
import numpy as np
import math

# Seq2Seq + Attention model


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


class Seq2SeqAttention(keras.Model):
    def __init__(self, latent_dim: int, dim_vectors: int, n_steps: int):
        super().__init__()

        self.dim_vectors = dim_vectors
        self.n_steps = n_steps

        # encoder LSTM
        self.encoder_lstm = keras.layers.LSTM(
            units=latent_dim, return_sequences=True, return_state=True
        )

        # decoder LSTM
        self.decoder_lstm = keras.layers.LSTM(
            units=latent_dim, return_sequences=True, return_state=True
        )

        # attention layer
        self.attention_layer = MultiplicativeAttention(latent_dim)

        # last dense layer for final output projection
        self.dense = keras.layers.Dense(dim_vectors)

    def call(self, inputs):
        encoder_inputs, decoder_inputs = inputs

        # encoder
        encoder_outputs, encoder_h, encoder_c = self.encoder_lstm(encoder_inputs)

        # decoder
        decoder_outputs, _, _ = self.decoder_lstm(
            decoder_inputs, initial_state=[encoder_h, encoder_c]
        )

        # attention mechanism
        context, _ = self.attention_layer([decoder_outputs, encoder_outputs])

        # concatenate attention output with decoder output
        decoder_concat_input = keras.layers.Concatenate(axis=-1)(
            [decoder_outputs, context]
        )

        # last dense layer
        outputs = self.dense(decoder_concat_input)
        return outputs

    def predict(self, X_test):
        encoder_outputs, state_h, state_c = self.encoder_lstm(X_test)
        # Starting decoder input with the last step of X_test
        last_values = X_test[:, -1:, :]
        pred_test = np.zeros((X_test.shape[0], self.n_steps, self.dim_vectors))

        for i in range(self.n_steps):
            decoder_outputs, state_h, state_c = self.decoder_lstm(
                last_values, initial_state=[state_h, state_c]
            )
            context, _ = self.attention_layer([decoder_outputs, encoder_outputs])
            output_seq = self.dense(
                keras.layers.Concatenate(axis=-1)([decoder_outputs, context])
            )
            # Append the prediction to the result
            pred_test[:, i, :] = output_seq[:, -1, :]
            # Update the decoder input with the new prediction
            last_values = output_seq[:, -1:, :]

        return pred_test


# Transformer based model


def positional_embedding(pos: int, k: int, d_model: int) -> float:
    i = k // 2  # k = 2*i + (k%2)

    if k % 2 == 0:
        return math.sin(pos / (10000 ** (2 * i / d_model)))
    else:
        return math.cos(pos / (10000 ** (2 * i / d_model)))


class ClassicPositionalEmbedding(keras.Layer):
    def __init__(self, max_length: int, d_model: int):
        super().__init__()

        self.max_length = max_length

        # creating positional encodings
        self.pos_encoding = tf.constant(
            [
                [positional_embedding(pos, k, d_model) for k in range(d_model)]
                for pos in range(max_length)
            ]
        )

    def call(self, inputs):
        # add positional encoding to input
        # we assume inputs have shape (batch_size, N, d_model), where N <= max_length
        # if inputs_length is None, then we assume inputs_length equals max_length
        N = inputs.shape[1]

        return inputs + self.pos_encoding[:N, :]


class FeedForward(tf.keras.Layer):
    def __init__(self, d_model: int, dff: int):
        super().__init__()

        # two linear transformations with a ReLU activation function between
        self.feedforward_net = keras.models.Sequential(
            [
                keras.layers.Dense(dff, activation="relu"),
                keras.layers.Dense(d_model),
            ]
        )

        # Residual connection
        self.add = tf.keras.layers.Add()

        # Layer Normalization
        self.layer_norm = tf.keras.layers.LayerNormalization()

    def call(self, inputs):
        x = self.feedforward_net(inputs)
        x = self.add([inputs, x])
        x = self.layer_norm(x)
        return x


class EncoderLayer(keras.Layer):
    def __init__(self, num_heads: int, dim_qkv: int, d_model: int, dff: int):
        super().__init__()

        # Multi-Head attention layer
        self.multi_head_attention = keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=dim_qkv, use_bias=False
        )

        # Residual connection
        self.add_layer = keras.layers.Add()

        # Layer Normalization
        self.layer_norm = keras.layers.LayerNormalization()

        # Feed-Forward Network
        self.ff_sublayer = FeedForward(d_model=d_model, dff=dff)

    def call(self, inputs):
        x = self.multi_head_attention(query=inputs, value=inputs)
        x = self.add_layer([inputs, x])
        x = self.layer_norm(x)
        x = self.ff_sublayer(x)

        return x


class TransformerEncoder(keras.Model):
    def __init__(
        self,
        num_heads: int,
        n_layers: int,
        sequence_length: int,
        d_model: int,
        dim_qkv: int,
        dff: int,
    ):
        super().__init__()

        self.num_head = num_heads
        self.n_layers = n_layers
        self.d_model = d_model
        self.dim_qkv = dim_qkv

        self.positional_encoding = ClassicPositionalEmbedding(
            max_length=sequence_length, d_model=d_model
        )
        self.encoder_layers = [
            EncoderLayer(num_heads=num_heads, dim_qkv=dim_qkv, d_model=d_model, dff=dff)
            for _ in range(n_layers)
        ]

    def call(self, inputs):
        x = self.positional_encoding(inputs)

        for i in range(self.n_layers):
            x = self.encoder_layers[i](x)

        return x


class CustomDecoderLayer(keras.Layer):
    def __init__(self, num_heads: int, dim_qkv: int, d_model: int, dff: int):
        super().__init__()

        # Multi-Head attention layer
        self.multi_head_attention = keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=dim_qkv, use_bias=False
        )

        # Residual connection
        self.add_layer = keras.layers.Add()

        # Layer Normalization
        self.layer_norm = keras.layers.LayerNormalization()

        # Feed-Forward
        self.ff_sublayer = FeedForward(d_model=d_model, dff=dff)

    def call(self, query, value):
        # query is the decoder layer input, while value is the output of the encoder

        x = self.multi_head_attention(query=query, value=value)
        x = self.add_layer([query, x])
        x = self.layer_norm(x)
        x = self.ff_sublayer(x)

        return x


class CustomDecoder(keras.Model):
    def __init__(
        self,
        num_heads: int,
        n_layers: int,
        sequence_length: int,
        d_model: int,
        dim_qkv: int,
        dff: int,
    ):
        super().__init__()

        self.num_head = num_heads
        self.n_layers = n_layers
        self.d_model = d_model
        self.dim_qkv = dim_qkv

        self.positional_encoding = ClassicPositionalEmbedding(
            max_length=sequence_length, d_model=d_model
        )
        self.decoder_layers = [
            CustomDecoderLayer(
                num_heads=num_heads, dim_qkv=dim_qkv, d_model=d_model, dff=dff
            )
            for _ in range(n_layers)
        ]

    def call(self, query, value):
        # query is the decoder input, while value is the output of the encoder
        x = self.positional_encoding(query)

        for i in range(self.n_layers):
            x = self.decoder_layers[i](x, value)

        return x


class TimeSeriesModel:
    def __init__(
        self,
        num_heads: int,
        n_layers: int,
        sequence_length: int,
        data_dim: int,
        d_model: int,
        dim_qkv: int,
        dff: int,
        pred_steps: int,
    ):
        self.pred_steps = pred_steps
        self.data_dim = data_dim

        # dense layer for data embedding
        self.input_embedding = keras.layers.Dense(d_model, use_bias=False)

        # dense layer to project model output in dimension data
        self.output_projection = keras.layers.Dense(data_dim, use_bias=False)

        # transformer encoder
        self.transformer_encoder = TransformerEncoder(
            num_heads=num_heads,
            n_layers=n_layers,
            sequence_length=sequence_length,
            d_model=d_model,
            dim_qkv=dim_qkv,
            dff=dff,
        )

        # custom decoder
        self.custom_decoder = CustomDecoder(
            num_heads=num_heads,
            n_layers=n_layers,
            sequence_length=sequence_length,
            d_model=d_model,
            dim_qkv=dim_qkv,
            dff=dff,
        )

        # pipeline definition
        encoder_inputs = keras.layers.Input(shape=(None, data_dim))
        decoder_inputs = keras.layers.Input(shape=(None, data_dim))

        encoder_inputs_embedded = self.input_embedding(encoder_inputs)
        encoder_output = self.transformer_encoder(encoder_inputs_embedded)

        decoder_inputs_embedded = self.input_embedding(decoder_inputs)
        output_decoder = self.custom_decoder(decoder_inputs_embedded, encoder_output)
        output_seq = self.output_projection(output_decoder)

        self.model = keras.Model([encoder_inputs, decoder_inputs], output_seq)

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
        encoder_input = self.input_embedding(X_test)
        encoder_output = self.transformer_encoder.predict(encoder_input)

        decoder_input = X_test[
            :, -1:, :
        ]  # decoder input first is the last step of the input

        pred_test = np.zeros(shape=(X_test.shape[0], self.pred_steps, self.data_dim))

        for i in range(self.pred_steps):
            decoder_input_embedded = self.input_embedding(
                decoder_input
            )  # data embedding
            output_decoder = self.custom_decoder(
                decoder_input_embedded, encoder_output
            )  # shape = (batch_size, decoder_input_length, d_model)
            output_seq = self.output_projection(
                output_decoder
            )  # shape = (batch_size, decoder_input_length, data_dim)
            pred_test[:, i, :] = output_seq[
                :, -1, :
            ]  # select only the predicted new step
            decoder_input = tf.concat([decoder_input, output_seq[:, -1:, :]], axis=1)

        return pred_test
