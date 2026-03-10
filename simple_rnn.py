import datetime
import sys
import typing as t

import tensorflow as tf

from smi_monitor import smi_summary, start_monitor, stop_monitor

tf.debugging.set_log_device_placement(True)  # prints every op + its device
# --- 1. Verify Devices ---
gpus = tf.config.list_physical_devices("GPU")
if len(gpus) < 2:
    print(
        f"Warning: Only {len(gpus)} GPU(s) detected. Code will fall back to CPU for missing devices."
    )
else:
    print(f"Detected {len(gpus)} GPUs: {gpus}")


DEVICE0 = "/GPU:0"
DEVICE1 = "/GPU:1"

# --- 2. Model Definition ---
@t.final
class MultiGPURNN(tf.keras.Model):  # pyright: ignore[reportMissingTypeArgument]
    def __init__(self):
        super(MultiGPURNN, self).__init__()
        # Explicit placement
        with tf.device("/GPU:0"):
            self.rnn_0 = tf.keras.layers.SimpleRNN(128, name="rnn_0")
        with tf.device("/GPU:1"):
            self.rnn_1 = tf.keras.layers.SimpleRNN(128, name="rnn_1")

        # self.classifier = layers.Dense(10, activation='softmax')
        with tf.device("/GPU:0"):
            self.classifier = tf.keras.layers.Dense(10, activation="softmax")

    # @t.override
    # def call(self, inputs):
    #    with tf.device('/GPU:0'):
    #        out0 = self.rnn_0(inputs)
    #    with tf.device('/GPU:1'):
    #        out1 = self.rnn_1(inputs)
    #    combined = tf.concat([out0, out1], axis=-1)
    #    return self.classifier(combined)

    @t.override
    def call(self, inputs):  # pyright: ignore[reportIncompatibleMethodOverride]
        # 2. Explicitly move/copy the input to both devices
        with tf.device("/GPU:0"):
            # This runs on GPU 0
            inputs_0 = tf.identity(inputs)
            out0 = self.rnn_0(inputs_0)

        with tf.device("/GPU:1"):
            # Force a copy of the inputs to GPU 1
            inputs_1 = tf.identity(inputs)
            out1 = self.rnn_1(inputs_1)

        # 3. Merge the results
        # TF will handle moving 'out1' from GPU 1 back to GPU 0 automatically
        with tf.device("/GPU:0"):
            combined = tf.concat([out0, out1], axis=-1)
            return self.classifier(combined)


# --- 2. Data Preparation ---
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0


# --- 4. Model Building ---
model = MultiGPURNN()
with tf.device("/GPU:0"):
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"], run_eagerly=False
    )

# --- 4. Profiling Setup ---
# We create a log directory with a timestamp
log_dir = "logs/profile/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# The TensorBoard callback handles the profiling
# 'profile_batch' tells it which batches to record (e.g., from batch 10 to 20)
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir, histogram_freq=1, profile_batch=(10, 20)
)

# --- 5. Training ---
print("Training and profiling...")
smi_stats, smi_stop, smi_thread = start_monitor()
model.fit(x_train, y_train, epochs=1, batch_size=64, callbacks=[tensorboard_callback])
stop_monitor(smi_stop, smi_thread)

smi_summary(smi_stats, "Simple RNN")
