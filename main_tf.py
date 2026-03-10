import time
import typing as t

import tensorflow as tf

from rnn_tf import build_model_keras_rnn, build_model_manual
from smi_monitor import smi_summary, smi_summary_stats, start_monitor, stop_monitor
from utils import print_avail_gpus, print_results, load_keras_imbdb_dataset

AnyFn: t.TypeAlias = t.Callable[[t.Any], t.Any]

# tf.debugging.set_log_device_placement(True)  # prints every op + its device
GPUS = tf.config.list_physical_devices("GPU")
for gpu in GPUS:
    tf.config.experimental.set_memory_growth(gpu, True)

# assert len(GPUS) >= 2

DEVICE_CPU = "/device:CPU:0"
DEVICE_GPU_0 = "/device:GPU:0"
DEVICE_GPU_1 = "/device:GPU:1"
PARAMS = {
    "VOCAB_SIZE": 10_000,
    "MAX_LEN": 100,
    "EMBED_DIM": 64,
    "RNN_UNITS": 64,
    "BATCH_SIZE": 128,
    "EPOCHS": 3,
}


def run_model(model: tf.keras.Model, label: str): # pyright: ignore[reportMissingTypeArgument]
    model.summary(line_length=70)
    (x_train, y_train), (x_test, y_test) = load_keras_imbdb_dataset(PARAMS)
    stats, stop_flag, mon_thread = start_monitor()
    t0 = time.time()
    history = model.fit(
        x_train,
        y_train,
        validation_split=0.1,
        batch_size=PARAMS["BATCH_SIZE"],
        epochs=PARAMS["EPOCHS"],
        verbose=1,
    )
    train_time = time.time() - t0
    stop_monitor(stop_flag, mon_thread)
    test_loss, test_acc = model.evaluate(x_test, y_test, batch_size=PARAMS["BATCH_SIZE"], verbose=0)
    result = {
        "label": label,
        "train_time": train_time,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "val_acc": history.history["val_accuracy"][-1],
        "gpu_stats": stats,
    }
    result['gpu'] = smi_summary_stats(stats)
    smi_summary(stats, label)
    del model
    tf.keras.backend.clear_session()
    return result
    #


def run_one_device():
    print_avail_gpus(GPUS)
    model = build_model_keras_rnn(
        PARAMS["RNN_UNITS"], DEVICE_GPU_0, DEVICE_GPU_0, DEVICE_GPU_0, **PARAMS
    )
    return run_model(model, "Keras RNN 1")


def run_manual_one_device():
    print_avail_gpus(GPUS)
    model = build_model_manual(
        PARAMS["RNN_UNITS"], DEVICE_GPU_0, DEVICE_GPU_0, DEVICE_GPU_0, **PARAMS
    )
    return run_model(model, "Manual RNN 1")

def run_manual_two_devices():
    print_avail_gpus(GPUS)
    model = build_model_manual(
        PARAMS["RNN_UNITS"], DEVICE_GPU_0, DEVICE_GPU_1, DEVICE_GPU_0, **PARAMS
    )
    return run_model(model, "Manual RNN 2")


def run_two_devices():
    print_avail_gpus(GPUS)
    model = build_model_keras_rnn(
        PARAMS["RNN_UNITS"], DEVICE_GPU_0, DEVICE_GPU_1, DEVICE_GPU_0, **PARAMS
    )
    return run_model(model, "Keras RNN 2")


def main():
    results = []
    results.append(run_one_device())
    results.append(run_manual_one_device())
    results.append(run_two_devices())
    results.append(run_manual_two_devices())
    print_results(results)


if __name__ == "__main__":
    main()
