import subprocess
import threading
import time
import typing as t
from collections import defaultdict

import numpy as np

# ═══════════════════════════════════════════════════════════════════
# GPU MONITOR
# ═══════════════════════════════════════════════════════════════════


def start_monitor(interval: float = 1.0):
    stats = []
    stop_flag = threading.Event()

    def _loop():
        while not stop_flag.is_set():
            try:
                r = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                ts = time.strftime("%H:%M:%S")
                for line in r.stdout.strip().splitlines():
                    p = [x.strip() for x in line.split(",")]
                    if len(p) == 4:
                        stats.append(
                            {
                                "time": ts,
                                "gpu": int(p[0]),
                                "util": int(p[1]),
                                "mem": int(p[2]),
                                "mem_tot": int(p[3]),
                            }
                        )
            except (FileNotFoundError, subprocess.TimeoutExpired):
                stop_flag.set()
                break
            time.sleep(interval)

    t = threading.Thread(target=_loop, daemon=True)
    t.start()
    return stats, stop_flag, t


def stop_monitor(stop_flag: threading.Event, thread: threading.Thread):
    stop_flag.set()
    thread.join(timeout=3)


def smi_summary_stats(stats: list[dict[str, t.Any]]):
    if not stats:
        return
    agg: dict[str, t.Any] = defaultdict(lambda: {"util": [], "mem": [], "mem_tot": 0})
    for r in stats:
        agg[r["gpu"]]["util"].append(r["util"])
        agg[r["gpu"]]["mem"].append(r["mem"])
        agg[r["gpu"]]["mem_tot"] = r["mem_tot"]
    rst = {}
    for g in sorted(agg.keys()):
        rst[g] = {}
        d = agg[g]
        rst[g]['util_avg'] = np.mean(d['util'])
        rst[g]['util_peak'] = max(d['util'])
        rst[g]['mem_avg'] = np.mean(d['mem'])
        rst[g]['mem_peak'] = max(d['mem'])
        rst[g]['mem_tot'] = d['mem_tot']
    return rst

def smi_summary(stats: list[dict[str, t.Any]], label: str):
    """Print per-GPU aggregate stats collected during one training run."""
    print(f"\n  ── GPU stats for {label} ──────────────────────────────")
    rst = smi_summary_stats(stats)
    if not rst:
        print("  (nvidia-smi unavailable — no hardware stats)")
        return
    for g in sorted(rst.keys()):
        d = rst[g]
        print(
            f"  GPU {g}:  util avg {d['util_avg']:5.1f}%  "
            + f"peak {d['util_peak']:3d}%  |  "
            + f"mem avg {d['mem_avg']:6.0f} MiB  "
            + f"peak {d['mem_peak']:5d}/{d['mem_tot']} MiB"
        )
