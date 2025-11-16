import pandas as pd
from tqdm import tqdm
import requests
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

GITHUB_API_BASE_URL = "https://patch-diff.githubusercontent.com/raw/"

MAX_THREADS = 128
MAX_PROCESSES = min(24, max(1, cpu_count() - 4)) 


def extract_diff_chunks(diff_text: str, context: int = 3):
    lines = diff_text.splitlines()
    n = len(lines)

    changed_line_indices = []

    for i, line in enumerate(lines):
        if line.startswith(("+++", "---", "diff --git")):
            continue
        if line.startswith("+") or line.startswith("-"):
            changed_line_indices.append(i)

    if not changed_line_indices:
        return []

    raw_chunks = []
    start = changed_line_indices[0]
    prev = start

    for idx in changed_line_indices[1:]:
        if idx == prev + 1:
            prev = idx
        else:
            raw_chunks.append((start, prev))
            start = idx
            prev = idx

    raw_chunks.append((start, prev))

    expanded = []
    for (s, e) in raw_chunks:
        new_s = max(0, s - context)
        new_e = min(n - 1, e + context)
        expanded.append([new_s, new_e])

    expanded.sort()
    merged_chunks = []
    cur_s, cur_e = expanded[0]

    for s, e in expanded[1:]:
        if s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            merged_chunks.append((cur_s, cur_e))
            cur_s, cur_e = s, e

    merged_chunks.append((cur_s, cur_e))

    final_chunks = []
    for (s, e) in merged_chunks:
        final_chunks.append({
            "start": s,
            "end": e,
            "lines": lines[s:e + 1]
        })

    return final_chunks


def build_output_string_from_chunks(chunks):
    output_lines = []
    for i, ch in enumerate(chunks, start=1):
        output_lines.append(f"=== CHUNK {i} (lines {ch['start']} - {ch['end']}) ===")
        for line in ch["lines"]:
            output_lines.append(line)
        output_lines.append("")
    return "\n".join(output_lines) + "\n"

def get_diff_text_from_github(html_url: str):
    if "https://github.com/" not in html_url:
        return None

    endpoint = html_url.replace("https://github.com/", "")
    gh_url = GITHUB_API_BASE_URL + f"{endpoint}.diff"

    try:
        response = requests.get(gh_url, timeout=5)
        if response.status_code == 200 and response.text.strip():
            return response.text
        return None
    except Exception:
        return None


def fetch_diff_one(pr_id: int, html_url: str, commit_dict: dict):

    diff_text = commit_dict.get(pr_id)
    if diff_text:
        return pr_id, diff_text

    diff_text = get_diff_text_from_github(html_url)
    return pr_id, diff_text

GLOBAL_DIFF_MAP = None


def init_worker(diff_map):
    global GLOBAL_DIFF_MAP
    GLOBAL_DIFF_MAP = diff_map


def parse_diff_one(pr_id: int):
    diff_text = GLOBAL_DIFF_MAP.get(pr_id)
    if not diff_text:
        return pr_id, "code diffs information not available"

    try:
        chunks = extract_diff_chunks(diff_text)
        if not chunks:
            return pr_id, "code diffs information not available"
        code_diff_str = build_output_string_from_chunks(chunks)
        return pr_id, code_diff_str
    except Exception:
        return pr_id, "code diffs information not available"

def main():
    pr_ids_df = pd.read_parquet("./output/pr_desc_title_combined_2025-11-10-22:27:17.parquet")
    matched_ids = set(pr_ids_df["matched_pr_ids"].tolist())
    print("matched_ids loaded.")

    pr_df = pd.read_parquet("hf://datasets/hao-li/AIDev/all_pull_request.parquet")
    print("pr_df loaded.")

    commit_df = pd.read_parquet("hf://datasets/hao-li/AIDev/pr_commit_details.parquet")
    print("commit_df loaded.")

    commit_df_nonnull = commit_df[commit_df["patch"].notna()]
    commit_dict = (
        commit_df_nonnull
        .groupby("pr_id", sort=False)["patch"]
        .apply(lambda x: "\n".join(x))
        .to_dict()
    )
    print(f"commit_dict built with {len(commit_dict)} PRs having patches.")

    filtered_pr_df = pr_df[pr_df["id"].isin(matched_ids)].copy()
    filtered_pr_df = filtered_pr_df.reset_index(drop=True)
    pr_ids = filtered_pr_df["id"].tolist()
    print(f"Processing {len(filtered_pr_df)} PRs out of {len(pr_df)} total.")

    diff_map: dict[int, str | None] = {}

    print("Starting threaded diff fetch (commit_dict + GitHub)...")
    with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        futures = {}
        for row in filtered_pr_df.itertuples(index=False):
            fut = executor.submit(fetch_diff_one, row.id, row.html_url, commit_dict)
            futures[fut] = row.id

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Fetching diffs"):
            pr_id, diff_text = fut.result()
            diff_map[pr_id] = diff_text

    print("Starting diff parsing")

    parsed_map: dict[int, str] = {}

    for pr_id in tqdm(pr_ids, desc="Parsing diffs"):
        diff_text = diff_map.get(pr_id)

        if not diff_text:
            parsed_map[pr_id] = "code diffs information not available"
            continue

        try:
            chunks = extract_diff_chunks(diff_text)
            if not chunks:
                parsed_map[pr_id] = "code diffs information not available"
            else:
                parsed_map[pr_id] = build_output_string_from_chunks(chunks)
        except Exception:
            parsed_map[pr_id] = "code diffs information not available"


    filtered_pr_df["code_diff"] = filtered_pr_df["id"].map(parsed_map)
    filtered_pr_df.to_parquet("matched_pr_with_diffs.parquet", index=False)
    print("Saved to matched_pr_with_diffs.parquet")



if __name__ == "__main__":
    main()
