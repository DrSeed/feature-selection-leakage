# Self-contained demo: labels are random, so any accuracy above ~50% is leakage.
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from leakage import leaky_selection_cv, honest_selection_cv, sample_level_cv, patient_level_cv


def noise_dataset(seed, n=60, genes=5000):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, genes)), rng.permutation(np.repeat([0, 1], n // 2))


def patient_dataset(seed, patients=30, per_patient=4, genes=200):
    rng = np.random.default_rng(seed)
    signature = rng.normal(0, 1.0, size=(patients, genes))
    pid = np.repeat(np.arange(patients), per_patient)
    X = signature[pid] + rng.normal(0, 0.7, size=(len(pid), genes))
    label = rng.permutation(np.repeat([0, 1], patients // 2))[pid]
    return X, label, pid


def main():
    os.makedirs("figures", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    rows = []
    for seed in range(15):
        X, y = noise_dataset(seed)
        rows.append({"experiment": "gene selection", "seed": seed,
                     "leaky": leaky_selection_cv(X, y), "honest": honest_selection_cv(X, y)})
        X, y, pid = patient_dataset(seed)
        rows.append({"experiment": "repeated patients", "seed": seed,
                     "leaky": sample_level_cv(X, y), "honest": patient_level_cv(X, y, pid)})
    df = pd.DataFrame(rows)
    df.to_csv("results/cv_runs.csv", index=False)
    summary = df.groupby("experiment")[["leaky", "honest"]].agg(["mean", "std"]).round(3)
    summary.columns = ["_".join(c) for c in summary.columns]
    summary.reset_index().to_csv("results/summary.csv", index=False)

    ks = [5, 20, 100, 500]
    X, y = noise_dataset(99)
    sweep = pd.DataFrame([{"k": k, "leaky": leaky_selection_cv(X, y, k=k), "honest": honest_selection_cv(X, y, k=k)} for k in ks])

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for ax, exp, labels in [(axes[0], "gene selection", ["select then CV", "select inside CV"]),
                            (axes[1], "repeated patients", ["split by sample", "split by patient"])]:
        sub = df[df["experiment"] == exp]
        ax.boxplot([sub["leaky"], sub["honest"]], widths=0.5)
        ax.set_xticks([1, 2], labels)
        ax.axhline(0.5, color="k", ls="--", lw=1, label="chance")
        ax.set_ylim(0.2, 1.05)
        ax.set_ylabel("cross-validated accuracy")
        ax.set_title(f"{exp.capitalize()}: labels are random")
        ax.legend()
    axes[2].plot(sweep["k"], sweep["leaky"], "o-", color="#c0504d", label="select then CV")
    axes[2].plot(sweep["k"], sweep["honest"], "o-", color="#2a7f62", label="select inside CV")
    axes[2].axhline(0.5, color="k", ls="--", lw=1)
    axes[2].set_xscale("log")
    axes[2].set_xlabel("genes kept (k) out of 5,000")
    axes[2].set_ylabel("accuracy")
    axes[2].set_title("Leakage at every k")
    axes[2].legend()
    fig.tight_layout()
    fig.savefig("figures/demo.png", dpi=120)
    print(summary)


if __name__ == "__main__":
    main()
