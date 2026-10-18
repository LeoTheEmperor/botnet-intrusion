"""
Load and label the N-BaIoT dataset for the Danmini Doorbell device.

Combines benign traffic + 4 Mirai attack files + 6 Gafgyt (BASHLITE) attack
files into one labeled DataFrame (11 classes, 115 numeric features).

Set the NBAIOT_DIR environment variable to the folder holding the 11 CSVs
(defaults to <repo>/data/raw).
"""
import os
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]      # repository root
RAW_DIR = Path(os.environ.get("NBAIOT_DIR", BASE / "data" / "raw"))
OUT_DIR = BASE / "data"

# filename -> (botnet family, attack subtype)
FILE_LABELS = {
    "benign_traffic.csv": ("benign", "benign"),
    "ack.csv": ("mirai", "ack"), "scan_1.csv": ("mirai", "scan"),
    "syn.csv": ("mirai", "syn"), "udp_1.csv": ("mirai", "udp"),
    "combo.csv": ("gafgyt", "combo"), "junk.csv": ("gafgyt", "junk"),
    "scan.csv": ("gafgyt", "scan"), "tcp.csv": ("gafgyt", "tcp"),
    "udp.csv": ("gafgyt", "udp"), "udpplain.csv": ("gafgyt", "udpplain"),
}


def load_labeled_dataset(sample_per_class=5000, random_state=42):
    """Load every file, tag rows with botnet/subtype/combined label, concatenate.
    sample_per_class caps rows per file (None = use everything)."""
    frames = []
    for fname, (botnet, subtype) in FILE_LABELS.items():
        df = pd.read_csv(RAW_DIR / fname)
        if sample_per_class is not None and len(df) > sample_per_class:
            df = df.sample(n=sample_per_class, random_state=random_state)
        df["botnet"] = botnet
        df["attack_subtype"] = subtype
        df["label"] = subtype if botnet == "benign" else f"{botnet}_{subtype}"
        frames.append(df)
        print(f"{fname:20s} -> {len(df):>7,} rows  [{botnet}/{subtype}]")
    full = pd.concat(frames, ignore_index=True)
    print(f"\nTotal rows: {len(full):,}   columns: {full.shape[1]}")
    print(full["label"].value_counts())
    return full


if __name__ == "__main__":
    OUT_DIR.mkdir(exist_ok=True)
    df = load_labeled_dataset(sample_per_class=5000)
    df.to_csv(OUT_DIR / "danmini_labeled_sample.csv", index=False)
    print(f"\nSaved to {OUT_DIR / 'danmini_labeled_sample.csv'}")
