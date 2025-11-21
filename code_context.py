import os
import pandas as pd
from tqdm import tqdm
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count

GITHUB_API_BASE_URL = "https://patch-diff.githubusercontent.com/raw/"

MAX_THREADS = 128
MAX_PROCESSES = min(24, max(1, cpu_count() - 4))


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

def split_diff_by_file(diff_text: str):
    lines = diff_text.splitlines()
    sections = []

    current_header = None
    current_lines = []

    for line in lines:
        if line.startswith("diff --git "):
            if current_header is not None:
                sections.append((current_header, current_lines))

            parts = line.strip().split()
            if len(parts) >= 4:
                a_path = parts[2]
                b_path = parts[3]
                current_header = f"{a_path} {b_path}"
            else:
                current_header = line.strip()

            current_lines = [line]
        else:
            if current_header is not None:
                current_lines.append(line)

    if current_header is not None:
        sections.append((current_header, current_lines))

    return sections


def extract_chunks_for_file(file_lines, context: int = 3):
    n = len(file_lines)
    if n == 0:
        return []

    changed_indices = []

    for i, line in enumerate(file_lines):
        if line.startswith(("diff --git", "index ")):
            continue
        if line.startswith("--- ") or line.startswith("+++ "):
            continue

        if line.startswith("+") or line.startswith("-"):
            changed_indices.append(i)

    if not changed_indices:
        return []

    raw_chunks = []
    start = changed_indices[0]
    prev = start

    for idx in changed_indices[1:]:
        if idx == prev + 1:
            prev = idx
        else:
            raw_chunks.append((start, prev))
            start = idx
            prev = idx
    raw_chunks.append((start, prev))

    expanded = []
    for s, e in raw_chunks:
        new_s = max(0, s - context)
        new_e = min(n - 1, e + context)
        expanded.append((new_s, new_e))

    expanded.sort()
    merged = []
    cur_s, cur_e = expanded[0]
    for s, e in expanded[1:]:
        if s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    merged.append((cur_s, cur_e))

    chunks = []
    for s, e in merged:
        chunks.append({
            "start": s,
            "end": e,
            "lines": file_lines[s:e + 1]
        })

    return chunks


def build_code_diff_for_pr(diff_text: str, context: int = 3):
    if not diff_text:
        return [], "code diffs information not available"

    sections = split_diff_by_file(diff_text)
    if not sections:
        return [], "code diffs information not available"

    files_changed = []
    output_lines = []

    for file_header, file_lines in sections:
        files_changed.append(file_header)

        chunks = extract_chunks_for_file(file_lines, context=context)
        if not chunks:
            continue

        output_lines.append(f"=== CHUNK {file_header} ===")

        for idx, ch in enumerate(chunks):
            for line in ch["lines"]:
                output_lines.append(line)
            if idx != len(chunks) - 1:
                output_lines.append("...")

        output_lines.append("")

    if not output_lines:
        return files_changed, "code diffs information not available"

    code_diff_str = "\n".join(output_lines).rstrip() + "\n"
    return files_changed, code_diff_str


def main():
    pr_ids_df = pd.read_parquet("./output/pr_desc_title_combined_2025-11-10-22:27:17.parquet")
    matched_ids = set(pr_ids_df["matched_pr_ids"].tolist())
    print(f"matched_ids loaded. {len(matched_ids)}")

    pr_df = pd.read_parquet("hf://datasets/hao-li/AIDev/all_pull_request.parquet")
    print(f"pr_df loaded. {len(pr_df)}")

    commit_df = pd.read_parquet("hf://datasets/hao-li/AIDev/pr_commit_details.parquet")
    print(f"commit_df loaded. {len(commit_df)}")

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

    files_changed_map: dict[int, list[str]] = {}
    code_diff_map: dict[int, str] = {}

    for pr_id in tqdm(pr_ids, desc="Parsing diffs"):
        diff_text = diff_map.get(pr_id)

        if not diff_text:
            files_changed_map[pr_id] = []
            code_diff_map[pr_id] = "code diffs information not available"
            continue

        try:
            files_changed, code_diff = build_code_diff_for_pr(diff_text, context=3)
            files_changed_map[pr_id] = files_changed
            code_diff_map[pr_id] = code_diff
        except Exception:
            files_changed_map[pr_id] = []
            code_diff_map[pr_id] = "code diffs information not available"

    filtered_pr_df["files_changed"] = filtered_pr_df["id"].map(files_changed_map)
    filtered_pr_df["code_diff"] = filtered_pr_df["id"].map(code_diff_map)

    filtered_pr_df.to_parquet("llm_matched_pr_with_diffs.parquet", index=False)
    print("Saved to matched_pr_with_diffs.parquet")

main()