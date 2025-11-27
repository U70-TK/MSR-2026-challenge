from transformers import AutoTokenizer
import pandas as pd
from tqdm import tqdm

df = pd.read_parquet("./matched_pr_with_diffs.parquet")
tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-120b")

df = df
exceeded = 0

for _, row in tqdm(df.iterrows(), total=len(df)):
    body = row.get("body","")
    code_diff = row.get("code_diff", "")
    text = ""
    if body:
        text += body
    if code_diff:
        text += code_diff
    if text:
        tokens = tokenizer.encode(text)
        if len(tokens) < 50000:
            exceeded += 1

print(exceeded)