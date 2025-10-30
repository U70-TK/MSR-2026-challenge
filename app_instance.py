import logging
from data_loader.keyword_loader import Keyword_loader
from data_loader.dataset_loader import Data_loader
from datetime import datetime
import os
import pandas as pd
from tqdm import tqdm
from constants.dataset_info import AIDev

class AppInstance():
    def __init__(self, huggingface_repo: str, keyword_dir: str, log_file_path:str, logger_id: str=datetime.now().strftime("%Y-%m-%d-%H:%M:%S")):
        self.log_file_path = log_file_path
        self.huggingface_repo = huggingface_repo
        self._logger: logging.Logger | None = None
        self._keyword_loader: Keyword_loader | None = None
        self._dataset_loader: Data_loader | None = None
        self.logger_id = logger_id
        self.keyword_dir = keyword_dir

    @property
    def logger(self) -> logging.Logger:
        if self._logger is not None:
            return self._logger
        os.makedirs(os.path.dirname(self.log_file_path), exist_ok=True)

        with open(self.log_file_path, "w") as f:
            f.write("")
    
        logger = logging.getLogger(self.logger_id)
        logger.setLevel(logging.INFO)
        fh = logging.FileHandler(self.log_file_path)
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        self._logger = logger
        return self._logger
    
    @property
    def keyword_loader(self):
        if self._keyword_loader is not None:
            return self._keyword_loader
        keyword_loader = Keyword_loader(
            keyword_dir_path=self.keyword_dir,
            logger=self.logger
        )
        self._keyword_loader = keyword_loader
        return self._keyword_loader
    
    @property
    def dataset_loader(self):
        if self._dataset_loader is not None:
            return self._dataset_loader
        dataset_loader = Data_loader(
            logger=self.logger,
            huggingface_repo=self.huggingface_repo
        )
        self._dataset_loader = dataset_loader
        return self._dataset_loader
        
    def match_any_regex(self, text: str):
        return self.keyword_loader.match_any(text)

    def _load_table_with_name(self, table_name):
        if hasattr(table_name, "value"):
            table_name = table_name.value

        method_name = f"get_{table_name}"
        if hasattr(self.dataset_loader, method_name):
            return getattr(self.dataset_loader, method_name)()
        else:
            self.logger.warning(f"No table called {table_name}")
            return None

    def match_pr_description(self):
        all_pr_request = self._load_table_with_name(AIDev.ALL_PULL_REQUEST)
        if all_pr_request is None or "body" not in all_pr_request.columns:
            self.logger.warning("ALL_PULL_REQUEST table not found or missing 'body' column.")
            return
        
        total_rows = len(all_pr_request)
        self.logger.info(f"Scanning {total_rows:,} PR descriptions")
        matched_records = []
        
        for _, row in tqdm(
            all_pr_request.iterrows(),
            total=total_rows,
            desc="Scanning PR descriptions",
            dynamic_ncols=True
        ):
            body = str(row["body"]) if pd.notna(row["body"]) else ""
            if self.keyword_loader.match_any(body):
                matched_records.append(row)

        matched_count = len(matched_records)
        self.logger.info(f"Found {matched_count:,} matching PRs out of {total_rows:,}")

        if matched_count > 0:
            matched_df = pd.DataFrame(matched_records)
            output_path = os.path.join(
                os.path.dirname(self.log_file_path),
                f"matched_prs_{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}.parquet"
            )
            matched_df.to_parquet(output_path, index=False)
            self.logger.info(f"Saved matched PRs to: {output_path}")
        else:
            self.logger.info("No matching PRs found.")