import re
import tarfile
import typing as t
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np

IMDB_URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
IMDB_NPZ_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/imdb.npz"
CACHE_DIR = Path("./imdb_cache")
CACHE_FILE = CACHE_DIR / "imdb_pytorch.pt"
KERAS_IMDB_CACHE_FILE = CACHE_DIR / 'imdb_keras_encoded.pt'


def print_avail_gpus(GPUS):
    print(f"\n{'═'*64}")
    print(f"  GPUs detected: {len(GPUS)}")
    for i, g in enumerate(GPUS):
        print(f"    [{i}] {g.name}")
    print(f"{'═'*64}\n")


def print_results(results: list[dict[str, t.Any]]):
    SEP = "─" * 80
    gpu_info = list(r["gpu"] for r in results if "gpu" in r)
    ngpus = max(len(gx) for gx in gpu_info) if gpu_info else 0
    if ngpus > 0:
        header = f"{'Model':<18} {'Test Acc':>9} {'Val Acc':>9} {'Time (s)':>9}"
        for gid in range(ngpus):
            header += f"{'GPU' + str(gid):>10}"
            header += f"{'GPU' + str(gid):>10}"
            header += f"{'GPU' + str(gid):>10}"
            header += f"{'GPU' + str(gid):>10}"
    else:
        header = f"{'Model':<48} {'Test Acc':>9} {'Val Acc':>9} {'Time (s)':>9}"

    print(f"\n\n{'═'*80}")
    print("  SUMMARY")
    print(f"{'═'*80}")
    print(f"  {header}")
    print(f"  {SEP}")
    for r in results:
        short = r["label"].split("—")[1].strip() if "—" in r["label"] else r["label"]
        if ngpus == 0:
            out_string = (
                f"  {short:<48} {r['test_acc']*100:>8.2f}%"
                + f" {r['val_acc']*100:>8.2f}%"
                + f" {r['train_time']:>9.1f}"
            )
        else:
            out_string = (
                f"  {short:<18} {r['test_acc']*100:>8.2f}%"
                + f" {r['val_acc']*100:>8.2f}%"
                + f" {r['train_time']:>9.1f}"
            )
            for _i, g in r["gpu"].items():
                # print(r["gpu"], g)
                out_string += (
                    f" {g['util_avg']:>8.1f}%"
                    + f" {g['util_peak']:>8.1f}%"
                    + f" {g['mem_avg']:>8.1f}M"
                    + f" {g['mem_peak']:>8.1f}M"
                )
        print(out_string)
    print(f"  {SEP}")


@t.final
def load_keras_imbdb_dataset(PARAMS: dict[str, t.Any]):  # pyright: ignore[a]
    import tensorflow as tf

    print("Loading IMDB dataset …")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(
        num_words=PARAMS["VOCAB_SIZE"]
    )
    x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=PARAMS["MAX_LEN"])
    x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=PARAMS["MAX_LEN"])
    print(f"  Train: {x_train.shape}   Test: {x_test.shape}\n")
    return (x_train, y_train), (x_test, y_test)


def tokenize(text: str):
    text = re.sub(r"<[^>]+>", " ", text.lower())
    return re.findall(r"[a-z']+", text)


def load_split(base, split):
    texts, labels = [], []
    for label_str, label_val in [("pos", 1), ("neg", 0)]:
        for fpath in sorted((base / split / label_str).glob("*.txt")):
            texts.append(fpath.read_text(encoding="utf-8"))
            labels.append(label_val)
    return texts, labels


def build_vocab(texts: list[str], max_vocab: int):
    counter = Counter()
    for tx in texts:
        counter.update(tokenize(tx))
    vocab = {w: i + 2 for i, (w, _) in enumerate(counter.most_common(max_vocab - 2))}
    vocab["<PAD>"] = 0
    vocab["<UNK>"] = 1
    return vocab


def encode_and_pad(texts, vocab, max_len: int):
    out = []
    for tx in texts:
        ids = [vocab.get(w, 1) for w in tokenize(tx)]
        ids = ids[:max_len] if len(ids) >= max_len else [0] * (max_len - len(ids)) + ids
        out.append(ids)
    return out


def fetch_imdb(PARAMS: dict[str, t.Any]):
    import torch

    CACHE_DIR.mkdir(exist_ok=True)
    if CACHE_FILE.exists():
        print("  Loading IMDB from cache ...")
        d = torch.load(CACHE_FILE)
        return (d['x_tr'], d['y_tr']), (d['x_te'], d['y_te'])

    tar_path = CACHE_DIR / "aclImdb_v1.tar.gz"
    if not tar_path.exists():
        print("  Downloading IMDB (~84 MB) ...")
        urllib.request.urlretrieve(IMDB_URL, tar_path)

    print("  Extracting ...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(CACHE_DIR)

    base = CACHE_DIR / "aclImdb"
    train_texts, train_labels = load_split(base, "train")
    test_texts, test_labels = load_split(base, "test")

    print("  Building vocab and encoding ...")
    vocab = build_vocab(train_texts, PARAMS["VOCAB_SIZE"])
    x_tr = torch.tensor(encode_and_pad(train_texts, vocab, PARAMS["MAX_LEN"]), dtype=torch.long)
    y_tr = torch.tensor(train_labels, dtype=torch.float32)
    x_te = torch.tensor(encode_and_pad(test_texts, vocab, PARAMS["MAX_LEN"]), dtype=torch.long)
    y_te = torch.tensor(test_labels, dtype=torch.float32)

    torch.save({"x_tr": x_tr, "y_tr": y_tr, "x_te": x_te, "y_te": y_te}, CACHE_FILE)
    print("  Saved to cache.")
    return (x_tr, y_tr), (x_te, y_te)


# Pad using pure numpy — same as Keras's pad_sequences
def pad_sequences(seqs, max_len):
    out = np.zeros((len(seqs), max_len), dtype=np.int64)
    for i, s in enumerate(seqs):
        s = np.array(s[-max_len:])  # truncate from left if too long
        out[i, -len(s):] = s  # pre-pad with zeros
    return out


def fetch_tflow_imdb(PARAMS: dict[str, t.Any]):
    import torch
    CACHE_DIR.mkdir(exist_ok=True)

    if CACHE_FILE.exists():
        print("  Loading Keras-encoded IMDB from cache ...")
        d = torch.load(CACHE_FILE, weights_only=True)
        return (d['x_tr'], d['y_tr']), (d['x_te'], d['y_te'])

    npz_path = CACHE_DIR / 'imdb_keras.npz'
    if not npz_path.exists():
        print("  Downloading Keras IMDB .npz (~17 MB) ...")
        urllib.request.urlretrieve(IMDB_NPZ_URL, npz_path)

    print("  Loading and padding sequences ...")
    data = np.load(npz_path, allow_pickle=True)

    # Replicate load_data(num_words=VOCAB_SIZE): map any index >= VOCAB_SIZE
    # to 1 (UNK), matching what Keras does when num_words is set.
    def filter_vocab(seqs):
        return [np.where(np.array(s) < PARAMS["VOCAB_SIZE"], s, 1) for s in seqs]

    x_tr_raw = filter_vocab(data['x_train'])
    x_te_raw = filter_vocab(data['x_test'])

    x_tr = torch.tensor(pad_sequences(x_tr_raw, PARAMS["MAX_LEN"]), dtype=torch.long)
    y_tr = torch.tensor(data['y_train'].astype(np.float32))
    x_te = torch.tensor(pad_sequences(x_te_raw, PARAMS["MAX_LEN"]), dtype=torch.long)
    y_te = torch.tensor(data['y_test'].astype(np.float32))

    torch.save({'x_tr': x_tr, 'y_tr': y_tr, 'x_te': x_te, 'y_te': y_te},
               KERAS_IMDB_CACHE_FILE)
    print("  Saved to cache.")
    return (x_tr, y_tr), (x_te, y_te)


def torch_data_loaders(PARAMS: dict[str, t.Any]):
    from torch.utils.data import DataLoader, TensorDataset

    # import torch
    # (x_tr, y_tr), (x_te, y_te) = load_keras_imbdb_dataset(PARAMS)
    #
    # Convert to PyTorch tensors (kept on CPU — DataLoader handles batching)
    # x_tr = torch.tensor(x_tr, dtype=torch.long)
    # y_tr = torch.tensor(y_tr, dtype=torch.float32)
    # x_te = torch.tensor(x_te, dtype=torch.long)
    # y_te = torch.tensor(y_te, dtype=torch.float32)

    #(x_tr, y_tr), (x_te, y_te) = fetch_imdb(PARAMS)
    (x_tr, y_tr), (x_te, y_te) = fetch_tflow_imdb(PARAMS)

    train_ds = TensorDataset(x_tr, y_tr)
    test_ds = TensorDataset(x_te, y_te)
    train_loader = DataLoader(train_ds, batch_size=PARAMS["BATCH_SIZE"], shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=PARAMS["BATCH_SIZE"])
    print(f"  Train: {x_tr.shape}   Test: {x_te.shape}\n")
    return train_loader, test_loader
