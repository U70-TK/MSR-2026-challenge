import math
import heapq
from transformers import AutoTokenizer
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import pyarrow as pa
import pyarrow.parquet as pq

# -------------------------------------------------
# Load tokenizer
# -------------------------------------------------
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-120b")

# -------------------------------------------------
# Math helpers (log-sum-exp + streaming median)
# -------------------------------------------------
def log_add(log_a, log_b):
    """Numerically stable log(exp(log_a) + exp(log_b))."""
    if log_a == -math.inf:
        return log_b
    if log_b == -math.inf:
        return log_a
    m = max(log_a, log_b)
    return m + math.log(math.exp(log_a - m) + math.exp(log_b - m))

# streaming median heaps
lower = []  # max heap (store negatives)
upper = []  # min heap

def add_number(x):
    """Insert number into streaming median."""
    if not lower or x <= -lower[0]:
        heapq.heappush(lower, -x)
    else:
        heapq.heappush(upper, x)

    if len(lower) > len(upper) + 1:
        heapq.heappush(upper, -heapq.heappop(lower))
    elif len(upper) > len(lower):
        heapq.heappush(lower, -heapq.heappop(upper))

def get_median():
    if len(lower) == len(upper):
        return (-lower[0] + upper[0]) / 2
    return -lower[0]

# -------------------------------------------------
# Function to compute token lengths + stats
# -------------------------------------------------
def compute_lengths_and_stats(df, label):
    log_sum = -math.inf
    count = 0

    min_tokens = math.inf
    max_tokens = 0
    count_over_128k = 0

    # reset median heaps
    global lower, upper
    lower, upper = [], []

    lengths = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {label}"):
        diff = row.get("code_diff", "")
        if not diff:
            diff = ""

        tokens = tokenizer.encode(diff)
        token_len = len(tokens)
        lengths.append(token_len)

        # accumulate log-sum-exp
        log_sum = log_add(log_sum, math.log(token_len))
        count += 1

        # min/max
        if token_len < min_tokens:
            min_tokens = token_len
        if token_len > max_tokens:
            max_tokens = token_len

        # >128k counter
        if token_len > 128000:
            count_over_128k += 1

        add_number(token_len)

    # average
    log_avg = log_sum - math.log(count)
    avg_tokens = math.exp(log_avg)

    # median
    median_tokens = get_median()

    # print out stats
    print("\n===== Statistics for", label, "=====")
    print(f"Number of diffs counted: {count}")
    print(f"Average token length: {avg_tokens:.4f}")
    print(f"Median token length: {median_tokens}")
    print(f"Smallest token length: {min_tokens}")
    print(f"Largest token length: {max_tokens}")
    print(f"Token lengths > 128k: {count_over_128k}")
    print("====================================\n")

    return lengths

# -------------------------------------------------
# Load both datasets
# -------------------------------------------------
print("Loading datasets...")
df_llm = pd.read_parquet("./data/llm_pr_with_diffs.parquet")
df_human = pd.read_parquet("./data/human_pr_with_diffs.parquet")

# -------------------------------------------------
# Compute everything
# -------------------------------------------------
llm_lengths = compute_lengths_and_stats(df_llm, "LLM PR Diffs")
human_lengths = compute_lengths_and_stats(df_human, "Human PR Diffs")

# -------------------------------------------------
# Save lengths to Parquet
# -------------------------------------------------
print("Saving length files...")
pq.write_table(pa.table({"token_length": llm_lengths}), "llm_token_lengths.parquet")
pq.write_table(pa.table({"token_length": human_lengths}), "human_token_lengths.parquet")

print("Saved:")
print("  llm_token_lengths.parquet")
print("  human_token_lengths.parquet")

# -------------------------------------------------
# Violin chart
# -------------------------------------------------
print("Generating violin chart...")

plt.figure(figsize=(10, 6))
plt.violinplot(
    [llm_lengths, human_lengths],
    showmeans=True,
    showmedians=True
)

plt.xticks([1, 2], ["LLM PR Diffs", "Human PR Diffs"])
plt.ylabel("Token Length")
plt.title("Distribution of Token Lengths in code_diff (LLM vs Human)")

plt.tight_layout()
plt.savefig("violin_llm_vs_human.png", dpi=300)
plt.close()

print("Chart saved as violin_llm_vs_human.png")
print("Done.")
