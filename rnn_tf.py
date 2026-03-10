import typing as t

import tensorflow as tf

AnyFn: t.TypeAlias = t.Callable[[t.Any], t.Any]


@t.final
class ManualLayer(tf.keras.layers.Layer):  # pyright: ignore[reportMissingTypeArgument]
    """Hand-rolled Elman RNN — all ops pinned to `device`."""

    def __init__(self, units: int, device: str, activation: str = "tanh", **kwargs: t.Any):
        super().__init__(**kwargs)
        self.units = units
        self.device = device
        self.activation = tf.keras.activations.get(activation)

    @t.override
    def build(self, input_shape: tuple[int]):
        d = input_shape[-1]
        with tf.device(self.device):
            self.W_x = self.add_weight(
                name="W_x", shape=(d, self.units), initializer="glorot_uniform"
            )
            self.W_h = self.add_weight(
                name="W_h", shape=(self.units, self.units), initializer="orthogonal"
            )
            self.b = self.add_weight(name="b", shape=(self.units,), initializer="zeros")
        super().build(input_shape)

    @t.override
    def call(self, inputs: t.Any):
        with tf.device(self.device):
            h = tf.zeros((tf.shape(inputs)[0], self.units))
            for tx in tf.range(tf.shape(inputs)[1]):
                h = self.activation(inputs[:, tx, :] @ self.W_x + h @ self.W_h + self.b)
            return h


@t.final
class TestRNNCell(tf.keras.layers.Layer):  # pyright: ignore[reportMissingTypeArgument]
    """
    A one-step RNN cell compatible with keras.layers.RNN.
    state_size / output_size are required by the RNN wrapper.
    """

    def __init__(self, units: int, activation: str = "tanh", **kwargs: t.Any):
        super().__init__(**kwargs)
        self.units: int = units
        self.state_size: int = units  # single hidden state vector
        self.output_size: int = units
        self.activation: AnyFn = tf.keras.activations.get(activation)

    @t.override
    def build(self, input_shape: tuple[int]):
        d = input_shape[-1]
        self.W_x = self.add_weight(name="W_x", shape=(d, self.units), initializer="glorot_uniform")
        self.W_h = self.add_weight(
            name="W_h", shape=(self.units, self.units), initializer="orthogonal"
        )
        self.b = self.add_weight(name="b", shape=(self.units,), initializer="zeros")
        super().build(input_shape)

    @t.override
    def call(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, inputs: t.Any, states: t.Any
    ):
        h_prev = states[0]
        h = self.activation(inputs @ self.W_x + h_prev @ self.W_h + self.b)
        return h, [h]  # (output, new_states)


class CopyToDevice(tf.keras.layers.Layer):  # pyright: ignore[reportMissingTypeArgument]
    """Explicit cross-device tensor transfer.

    Using a proper Layer subclass (rather than Lambda) is important because
    the Keras Functional API traces the graph through Layer objects. Lambda
    layers with tf.device inside are not reliably honoured during graph
    construction across all TF versions — the device context can be ignored
    or overridden when TF replays the graph.

    A Layer subclass bypasses this: TF sees it as an opaque node with a
    declared device, and inserts the MemcpyH2D / MemcpyD2H op cleanly.

    Usage:
        emb0 = CopyToDevice(DEVICE_0, name='copy_to_gpu0')(emb)
    """

    def __init__(self, device: str, **kwargs: t.Any):
        super().__init__(**kwargs)
        self.target_device: str = device

    def call(self, inputs: t.Any):
        with tf.device(self.target_device):
            return tf.identity(inputs)

    @t.override
    def get_config(self):
        config = super().get_config()
        config.update({"device": self.target_device})
        return config


def copy_to(device: str, layer_name: str):
    return CopyToDevice(device, name=layer_name)


def build_output_head(b1, b2, out_device: str):
    """Shared merge + classification head used by all three models.
    Dense is constructed *inside* tf.device so its kernel/bias variables
    are created on CPU — constructing outside and calling inside is not
    enough, the variables would still land on the default GPU.
    """
    b1_cpu = copy_to(out_device, "b1_to_cpu")(b1)
    b2_cpu = copy_to(out_device, "b2_to_cpu")(b2)
    with tf.device(out_device):
        merged = tf.keras.layers.Concatenate(name="merge")([b1_cpu, b2_cpu])
        drop = tf.keras.layers.Dropout(0.4)(merged)
        dense = tf.keras.layers.Dense(1, activation="sigmoid", name="output")
        out = dense(drop)
    return out


# ══════════════════════════════════════════════════════════════════════
# Model A — ManualRNNLayer (hand-rolled Elman RNN)
# ══════════════════════════════════════════════════════════════════════
def build_model_manual(
    units: int, device0: str, device1: str, out_device: str, **params: t.Any
) -> tf.keras.Model:  # pyright: ignore[reportMissingTypeArgument]
    """Both RNN branches use ManualRNNLayer, which pins all ops internally
    via tf.device(self.device) inside build() and call(). No copy_to is
    needed before the RNN layers because ManualRNNLayer reads its input
    under its own device context, letting TF insert the H2D copy itself.
    """
    inputs = tf.keras.Input(shape=(params["MAX_LEN"],), dtype=tf.int32, name="tokens")
    emb = tf.keras.layers.Embedding(params["VOCAB_SIZE"], params["EMBED_DIM"], name="embedding")(
        inputs
    )

    # Branch 1 on GPU 0 — tanh activation
    b1 = ManualLayer(units, device0, activation="tanh", name="rnn_b1")(emb)
    # Branch 2 on GPU 1 — relu activation
    b2 = ManualLayer(units, device1, activation="relu", name="rnn_b2")(emb)

    out = build_output_head(b1, b2, out_device)
    model = tf.keras.Model(inputs, out, name="Model_A_Manual")
    model.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy"]
    )
    return model


# ══════════════════════════════════════════════════════════════════════
# Model C — tf.keras.layers.RNN with a custom SimpleRNNCell
# ══════════════════════════════════════════════════════════════════════
def build_model_keras_rnn(
    units: int, device0: str, device1: str, out_device: str, **params: t.Any
) -> tf.keras.Model: # pyright: ignore[reportMissingTypeArgument]
    """keras.layers.RNN is the generic loop wrapper; the cell defines one step.
    The same two placement rules as Model B apply, with one addition:
    the *cell* must also be constructed inside tf.device because it is a
    separate Layer with its own W_x/W_h/b weights, independent of the
    outer RNN wrapper.
    """
    inputs = tf.keras.Input(shape=(params["MAX_LEN"],), dtype=tf.int32, name="tokens")
    emb = tf.keras.layers.Embedding(params["VOCAB_SIZE"], params["EMBED_DIM"], name="embedding")(
        inputs
    )

    # Branch 1 on GPU 0 — tanh activation
    emb0 = copy_to(device0, "copy_to_gpu0")(emb)
    with tf.device(device0):
        cell_b1 = TestRNNCell(units, activation="tanh", name="cell_b1")
        b1 = tf.keras.layers.RNN(cell_b1, name="kerasrnn_b1")(emb0)

    # Branch 2 on GPU 1 — relu activation
    emb1 = copy_to(device1, "copy_to_gpu1")(emb)
    with tf.device(device1):
        cell_b2 = TestRNNCell(units, activation="relu", name="cell_b2")
        b2 = tf.keras.layers.RNN(cell_b2, name="kerasrnn_b2")(emb1)

    out = build_output_head(b1, b2, out_device)
    model = tf.keras.Model(inputs, out, name="Model_C_Keras_RNN")
    model.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy"]
    )
    return model
