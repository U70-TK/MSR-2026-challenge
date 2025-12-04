import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

llm_df = pd.read_parquet("llm_token_lengths.parquet")
human_df = pd.read_parquet("human_token_lengths.parquet")

llm_lengths = llm_df["token_length"].to_numpy()
human_lengths = human_df["token_length"].to_numpy()

llm_cap = np.percentile(llm_lengths, 95)
human_cap = np.percentile(human_lengths, 95)

print(f"LLM 95th percentile threshold: {llm_cap}")
print(f"Human 95th percentile threshold: {human_cap}")

llm_top5 = llm_lengths[llm_lengths >= llm_cap]
human_top5 = human_lengths[human_lengths >= human_cap]

print(f"LLM top 5% count: {len(llm_top5)}")
print(f"Human top 5% count: {len(human_top5)}")

plt.figure(figsize=(10, 6))

plt.violinplot(
    [llm_top5, human_top5],
    showmeans=True,
    showmedians=True
)

plt.xticks([1, 2], ["LLM PR Diffs (Top 5%)", "Human PR Diffs (Top 5%)"])
plt.ylabel("Token Length")
plt.title("Distribution of Token Lengths (Top 5% Only)")

plt.tight_layout()
plt.savefig("violin_llm_vs_human_top5.png", dpi=300)
plt.close()

print("Saved violin_llm_vs_human_top5.png")
