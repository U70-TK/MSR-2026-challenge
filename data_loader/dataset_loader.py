import pandas as pd
import os
from constants.dataset_info import AIDev

class Data_loader:
    def __init__(self, logger, huggingface_repo: str):
        self.logger = logger
        self.data_dir = huggingface_repo
        self._cache: dict[str, pd.DataFrame] = {}

    def _load_table(self, name: str):
        if name in self._cache:
            return self._cache[name]

        parquet_file = f"{name}.parquet"
        dataset_dir = self.data_dir + parquet_file

        try:
            self.logger.info(f"{'-'*10} Loading {name} {'-'*10}")
            df = pd.read_parquet(dataset_dir)
            self._cache[name] = df
            self.logger.info(f"{'-'*10} Loaded {name} ({len(df):,} rows) {'-'*10}")
            return df
        except Exception as e:
            self.logger.error(f"Cannot load {dataset_dir}: {e}")
            return None

    def get_all_pull_request(self):
        return self._load_table(AIDev.ALL_PULL_REQUEST.value)

    def get_all_repository(self):
        return self._load_table(AIDev.ALL_REPOSITORY.value)

    def get_all_user(self):
        return self._load_table(AIDev.ALL_USER.value)

    def get_pr_commits(self):
        return self._load_table(AIDev.PR_COMMITS.value)

    def get_pr_commit_details(self):
        return self._load_table(AIDev.PR_COMMIT_DETAILS.value)

    def get_pr_reviews(self):
        return self._load_table(AIDev.PR_REVIEWS.value)

    def get_pr_timeline(self):
        return self._load_table(AIDev.PR_TIMELINE.value)

    def clear_cache(self):
        self._cache.clear()
        self.logger.info("Cache cleared.")
