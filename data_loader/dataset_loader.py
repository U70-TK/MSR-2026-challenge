from datasets import load_dataset
import pandas as pd
from constants.dataset_info import AIDev

class Data_loader:
    def __init__(self, logger, huggingface_repo: str):
        self.logger = logger
        self.repo = huggingface_repo.rstrip("/")
        self._cache = {}

    def _load_table(self, name: str) -> pd.DataFrame | None:
        if name in self._cache:
            return self._cache[name]

        try:
            self.logger.info(f"Loading table '{name}' from {self.repo}")
            dataset = load_dataset(self.repo, data_files=f"{name}.parquet", split="train")
            df = dataset.to_pandas()
            self._cache[name] = df
            self.logger.info(f"Loaded {name} ({len(df):,} rows)")
            return df
        except Exception as e:
            self.logger.error(f"Cannot load {name} from {self.repo}: {e}")
            return None
        
    def get_all_pull_request(self):
        return self._load_table(AIDev.ALL_PULL_REQUEST.value)
    
    def get_human_pull_request(self):
        return self._load_table(AIDev.HUMAN_PULL_REQUEST.value)

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
