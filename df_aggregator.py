import pandas as pd
import os
from typing import List

def merge_parquets_with_prefix(root_path: str,
                               prefix: str,
                               output_filename: str) -> pd.DataFrame:
    all_files: List[str] = [
        os.path.join(root_path, f)
        for f in os.listdir(root_path)
        if f.startswith(prefix) and f.endswith(".parquet")
    ]

    if not all_files:
        raise FileNotFoundError(f"No parquet files found with prefix '{prefix}' in {root_path}")

    print(f"Found {len(all_files)} matching parquet files:")
    for f in all_files:
        print(" -", f)

    dfs = [pd.read_parquet(fp) for fp in all_files]

    combined_df = pd.concat(dfs, ignore_index=True)

    if "id" in combined_df.columns:
        combined_df.drop_duplicates(subset=["id"], inplace=True)
    else:
        print("WARNING: 'id' column not found; skipping deduplication.")

    out_path = os.path.join(root_path, output_filename)
    combined_df.to_parquet(out_path, index=False)

    print(f"\nMerged and deduplicated parquet saved to:\n  {out_path}")
    print(f"Final row count: {len(combined_df)}")

    return combined_df


if __name__ == "__main__":
    merge_parquets_with_prefix(
        root_path="/home/t577wang/MSR-2026-challenge/output",
        prefix="all_pull_request_cwe_output",
        output_filename="AIDev.ALL_PULL_REQUEST.parquet"
    )
