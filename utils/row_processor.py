import pandas as pd
from collections import Counter

class RowProcessors:

    @staticmethod
    def process_pr_description(args):
        row, compiled_regex_lst = args
        body = str(row.get("body", "")) if pd.notna(row.get("body", "")) else ""
        local_counts = Counter()
        matched_id = None

        for pattern, regex_str in compiled_regex_lst:
            if pattern.search(body):
                matched_id = row["id"]
                local_counts[regex_str] += 1

        return matched_id, local_counts

    @staticmethod
    def process_pr_commit_message(args):
        row, compiled_regex_lst = args
        msg = str(row.get("message", "")) if pd.notna(row.get("message", "")) else ""
        local_counts = Counter()
        matched_id = None

        for pattern, regex_str in compiled_regex_lst:
            if pattern.search(msg):
                matched_id = row["pr_id"]
                local_counts[regex_str] += 1

        return matched_id, local_counts
