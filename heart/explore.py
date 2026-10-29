"""Generate exploratory figures using only the fixed training partition."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from heart.config import NUMERIC, REPORT_DIR
from heart.data import load_dataset, split_dataset


def main():
    X, y = load_dataset()
    train, _, labels, _ = split_dataset(X, y)
    directory = REPORT_DIR / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    labels.value_counts().sort_index().rename(index={0: "Absent", 1: "Present"}).plot.bar(
        ax=axes[0], color=["#18786F", "#D68A35"], rot=0, title="Training target counts")
    train.isna().sum().plot.bar(ax=axes[1], color="#18786F", title="Training missing values")
    axes[0].set_ylabel("Records")
    axes[1].set_ylabel("Missing entries")
    fig.tight_layout()
    fig.savefig(directory / "training_audit.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for name, ax in zip(NUMERIC, axes.flat):
        for label, color in [(0, "#18786F"), (1, "#D68A35")]:
            ax.hist(train.loc[labels == label, name].dropna(), bins=12, alpha=0.55,
                    label="Absent" if label == 0 else "Present", color=color)
        ax.set_title(name)
        ax.set_ylabel("Records")
    axes[0, 0].legend()
    fig.suptitle("Numeric distributions by target (training partition only)")
    fig.tight_layout()
    fig.savefig(directory / "training_distributions.png", dpi=160)
    plt.close(fig)
    train.describe().to_csv(REPORT_DIR / "training_summary.csv")
    print("Saved training-only exploration figures and summary.")


if __name__ == "__main__":
    main()
