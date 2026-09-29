from pathlib import Path

import matplotlib.pyplot as plt


OUTPUT_DIR = Path("evaluation/graphs")
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# -----------------------------
# 실험 결과
# -----------------------------

models = [
    "MiniLM",
    "Multilingual E5"
]

recall = [
    66.7,
    83.3
]

latency = [
    15.74,
    34.07
]


# -----------------------------
# Recall@3 Graph
# -----------------------------

plt.figure(
    figsize=(7, 5)
)

bars = plt.bar(
    models,
    recall
)

plt.title(
    "Retriever Recall@3 Comparison"
)

plt.ylabel(
    "Recall@3 (%)"
)

plt.ylim(
    0,
    100
)

for bar, value in zip(
    bars,
    recall
):
    plt.text(
        bar.get_x()
        + bar.get_width() / 2,
        value + 2,
        f"{value:.1f}%",
        ha="center"
    )

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "recall_comparison.png",
    dpi=200
)

plt.close()


# -----------------------------
# Latency Graph
# -----------------------------

plt.figure(
    figsize=(7, 5)
)

bars = plt.bar(
    models,
    latency
)

plt.title(
    "Retriever Latency Comparison"
)

plt.ylabel(
    "Average Latency (ms)"
)

for bar, value in zip(
    bars,
    latency
):
    plt.text(
        bar.get_x()
        + bar.get_width() / 2,
        value + 1,
        f"{value:.2f} ms",
        ha="center"
    )

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "latency_comparison.png",
    dpi=200
)

plt.close()


print(
    "그래프 생성 완료:"
)

print(
    OUTPUT_DIR
    / "recall_comparison.png"
)

print(
    OUTPUT_DIR
    / "latency_comparison.png"
)