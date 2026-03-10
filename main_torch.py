import os
import time
import typing as t
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from rnn_torch import CustomCellModel, ManualRNNModel, evaluate, train_epoch
from smi_monitor import (smi_summary, smi_summary_stats, start_monitor,
                         stop_monitor)
from utils import print_results, torch_data_loaders

DEVICE_CPU = "/device:CPU:0"
DEVICE_GPU_0 = "cuda:0"
DEVICE_GPU_1 = "cuda:1"
PARAMS = {
    "VOCAB_SIZE": 10_000,
    "MAX_LEN": 100,
    "EMBED_DIM": 64,
    "RNN_UNITS": 64,
    "BATCH_SIZE": 128,
    "EPOCHS": 3,
    "LOG_DIR": "run_log/"
}

N_GPUS = torch.cuda.device_count()
print(N_GPUS)
assert N_GPUS >= 2
GPU_DEVICE_0 = torch.device("cuda:0")
GPU_DEVICE_1 = torch.device("cuda:1")

# Output head always on DEV0 (or CPU) — needs both branch outputs
OUTPU_DEVICE = GPU_DEVICE_0


def print_avail_gpus():
    N_GPUS = torch.cuda.device_count()
    print(f"\n{'═'*64}")
    print(f"  GPUs detected: {N_GPUS}")
    for i in range(N_GPUS):
        print(f"    [{i}] {torch.cuda.get_device_name(i)}")
    print(f"{'═'*64}\n")


def run_model(
    model,
    label: str,
    loaders: tuple[DataLoader, DataLoader],  # pyright: ignore[reportMissingTypeArgument]
    params: dict[str, t.Any],
):
    train_loader, test_loader = loaders
    out_device = model.out_device
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    run_dir = os.path.join(params["LOG_DIR"], label.replace(' ', '_'), datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    epochs = params["EPOCHS"]
    smi_stats, smi_stop, smi_thread = start_monitor()
    writer = SummaryWriter(run_dir)
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, out_device)
        val_loss, val_acc = evaluate(model, test_loader, out_device)

        # Log scalars to TensorBoard
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Acc/train", train_acc, epoch)
        writer.add_scalar("Acc/val", val_acc, epoch)

        # Log weight histograms
        for name, param in model.named_parameters():
            writer.add_histogram(name, param.data.cpu(), epoch)

        print(
            f"  Epoch {epoch}/{epochs}  "
            + f"loss {train_loss:.4f}  acc {train_acc*100:.2f}%  |  "
            + f"val_loss {val_loss:.4f}  val_acc {val_acc*100:.2f}%"
        )
    elapsed = time.time() - t0
    stop_monitor(smi_stop, smi_thread)
    test_loss, test_acc = evaluate(model, test_loader, model.out_device)
    writer.add_hparams(
        hparam_dict={
            "model": str(name),
            "lr": float(1e-3),
            "batch_size": int(PARAMS["BATCH_SIZE"]),
        },
        metric_dict={
            "hparam/test_acc": float(test_acc),
            "hparam/test_loss": float(test_loss),
        },
    )
    writer.close()
    smi_summary(smi_stats, name)
    gstats = smi_summary_stats(smi_stats)
    result = {
        "label": label,
        "test_acc": test_acc,
        "test_loss": test_loss,
        "train_time": elapsed,
        "run_dir": run_dir,
        "val_acc": val_acc,
        "gpu": gstats,
        "gpu_stats": smi_stats,
    }
    del model
    torch.cuda.empty_cache()
    return result


def run_one_device():
    model = CustomCellModel(PARAMS["RNN_UNITS"], DEVICE_GPU_0, DEVICE_GPU_0, DEVICE_GPU_0, PARAMS)
    return run_model(model, "Custom RNN", torch_data_loaders(PARAMS), PARAMS)


def run_two_devices():
    model = CustomCellModel(PARAMS["RNN_UNITS"], DEVICE_GPU_0, DEVICE_GPU_1, DEVICE_GPU_0, PARAMS)
    return run_model(model, "Custom RNN 2", torch_data_loaders(PARAMS), PARAMS)


def run_manual_one_device():
    model = ManualRNNModel(PARAMS["RNN_UNITS"], DEVICE_GPU_0, DEVICE_GPU_0, DEVICE_GPU_0, PARAMS)
    return run_model(model, "Manual RNN", torch_data_loaders(PARAMS), PARAMS)


def run_manual_two_devices():
    model = ManualRNNModel(PARAMS["RNN_UNITS"], DEVICE_GPU_0, DEVICE_GPU_1, DEVICE_GPU_0, PARAMS)
    return run_model(model, "Manual RNN 2", torch_data_loaders(PARAMS), PARAMS)


def main():
    print_avail_gpus()
    results = []
    results.append(run_one_device())
    results.append(run_manual_one_device())
    results.append(run_two_devices())
    results.append(run_manual_two_devices())
    print_results(results)


if __name__ == "__main__":
    main()
