import tensorflow as tf
from tensorflow.keras import Model, layers
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence

from smi_monitor import smi_summary, start_monitor, stop_monitor

# 1. Hyperparameters & Data Preprocessing
MAX_FEATURES = 10000  # Number of words to consider as features
MAXLEN = 500  # Cut texts after this number of words
BATCH_SIZE = 64

DEVICE0 = "/GPU:0"
DEVICE1 = "/GPU:1"
# DEVICE1 = DEVICE0
# DATASET = "imdb"
DATASET = "mnist"


if DEVICE1 != DEVICE0:
    tf.debugging.set_log_device_placement(True)  # prints every op + its device
    #tf.config.set_soft_device_placement(True)

print("Loading data...")
if DATASET == "imdb":
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=MAX_FEATURES)
    x_train = sequence.pad_sequences(x_train, maxlen=MAXLEN)
    x_test = sequence.pad_sequences(x_test, maxlen=MAXLEN)
else:
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0


optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=1e-3)

class ManualMultiGPURNN(Model): # pyright: ignore[reportMissingTypeArgument]
    def __init__(self, max_features, maxlen):
        super(ManualMultiGPURNN, self).__init__()
        # Layer setup remains the same
        with tf.device(DEVICE0):
            self.embedding0 = layers.Embedding(max_features, 128)
            self.rnn_0 = layers.SimpleRNN(128)
        with tf.device(DEVICE1):
            self.embedding1 = layers.Embedding(max_features, 128)
            self.rnn_1 = layers.SimpleRNN(128)
        with tf.device(DEVICE0):
            self.classifier = layers.Dense(1, activation="sigmoid")

    def call(self, inputs, training=False):  # pyright: ignore[reportIncompatibleMethodOverride]
        # Forward pass logic
        with tf.device(DEVICE0):
            inputs0 = tf.identity(inputs)
            if DATASET == "imdb":
                x0 = self.embedding0(inputs0)
            else:
                x0 = inputs0
            out0 = self.rnn_0(x0, training=training)

        with tf.device(DEVICE1):
            # Ensure GPU:1 has its own copy of data
            inputs1 = tf.identity(inputs)
            if DATASET == "imdb":
                x1 = self.embedding1(inputs1)
            else:
                x1 = inputs1
            out1 = self.rnn_1(x1, training=training)

        with tf.device(DEVICE0):
            combined = tf.concat([out0, out1], axis=-1)
            return self.classifier(combined)

    # Overriding train_step gives us manual control over the loop
    @tf.function
    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            # 1. Forward pass (uses our device-aware call method)
            y_pred = self(x, training=True)

            # 2. Loss calculation (explicitly on GPU 0)
            with tf.device(DEVICE0):
                loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        # 3. Gradient computation
        # Note: Tape usually handles cross-device gradients automatically
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.compiled_metrics.update_state(y, y_pred)

        # 4. Optimization (explicitly on GPU 0 to avoid resource errors)
        #for grad, var in zip(gradients, self.trainable_variables):
        #    if grad is not None:
        #        with tf.device(var.device):
        #            self.optimizer.apply_gradients([(grad, var)])

        #with tf.device(DEVICE0):
        #    self.compiled_metrics.update_state(y, y_pred)

        return {m.name: m.result() for m in self.metrics}


# Initialize model
model = ManualMultiGPURNN(10000, 500)

# 3. Compile and Train
#with tf.device(DEVICE0):
model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])

#optimizer.build(model.trainable_variables)
# Create dummy data to trigger variable initialization
#dummy_x = tf.zeros((1, 500))
#dummy_y = tf.zeros((1, 1))

# Run one train step manually to "seat" the optimizer variables
#with tf.device(DEVICE0):
#    model.train_on_batch(dummy_x, dummy_y)

print("Optimizer slots initialized. Check logs for placement now.")
#print("Starting training with IMDB on 2 GPUs...")
smi_stats, smi_stop, smi_thread = start_monitor()
model.fit(x_train, y_train, epochs=3, batch_size=BATCH_SIZE, validation_data=(x_test, y_test))
stop_monitor(smi_stop, smi_thread)
smi_summary(smi_stats, "Simple Manual RNN")
#print("Hello")
