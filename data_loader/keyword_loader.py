import re
from tqdm import tqdm
import os
from collections import defaultdict

class Keyword_loader:
    def __init__(self, keyword_dir_path: str, logger, extension: str = ".txt"):
        self.keyword_dir_path = keyword_dir_path
        self.extension = extension
        self.file_path_lst = []
        self.compiled_regex_lst = []
        self._find_all_files()
        self.logger = logger
        self.regex_match_counts = defaultdict(int)

    def _find_all_files(self):
        for root, _, files in os.walk(self.keyword_dir_path):
            for filename in files:
                if filename.endswith(self.extension):
                    self.file_path_lst.append(os.path.join(root, filename))

    def _get_all_files(self):
        return self.file_path_lst

    def _load_keywords(self):
        self.logger.info(f"{'-'*10} Loading keyword list {'-'*10}")
        progress_bar = tqdm(total=len(self.file_path_lst))
        regex_strings = set()

        for file_path in self.file_path_lst:
            with open(file_path, "r") as file:
                for line in file:
                    regex_str = line.strip()
                    if not regex_str or regex_str.startswith("->"):
                        continue
                    regex_strings.add(regex_str)
            progress_bar.update(1)

        progress_bar.close()
        self.logger.info(f"Collected {len(regex_strings)} unique regex patterns")

        for regex_str in regex_strings:
            try:
                pattern = re.compile(regex_str, re.IGNORECASE)
                self.compiled_regex_lst.append((pattern, regex_str))
                self.regex_match_counts[regex_str] = 0
            except re.error as e:
                self.logger.warning(f"Skipped invalid regex: '{regex_str}' ({e})")

        self.logger.info(f"Successfully compiled {len(self.compiled_regex_lst)} regex patterns.")
        self.logger.info(f"{'-'*10} Loading keyword finished {'-'*10}")

    def match_any(self, text: str) -> bool:
        if not self.compiled_regex_lst:
            self._load_keywords()
            if not self.compiled_regex_lst:
                self.logger.warning("Error: Compiled keyword list is empty")
                return False

        matched = False
        for pattern, regex_str in self.compiled_regex_lst:
            if pattern.search(text):
                self.regex_match_counts[regex_str] += 1
                matched = True
        return matched

    def log_regex_statistics(self):
        if not self.regex_match_counts:
            self.logger.info("No regex matches recorded.")
            return

        total_matches = sum(self.regex_match_counts.values())
        self.logger.info(f"{'-'*10} Regex Match Summary (Total {total_matches:,}) {'-'*10}")

        sorted_counts = sorted(self.regex_match_counts.items(), key=lambda x: x[1], reverse=True)
        for regex, count in sorted_counts:
            self.logger.info(f"'{regex}': {count} matches")

        self.logger.info(f"{'-'*10} End of Regex Match Summary {'-'*10}")
