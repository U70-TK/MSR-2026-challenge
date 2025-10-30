import logging
from data_loader.keyword_loader import Keyword_loader
from data_loader.dataset_loader import Data_loader
from data_writer.df_writer import DF_writer
from datetime import datetime
import os
import pandas as pd
from tqdm import tqdm
from constants.dataset_info import AIDev

class AppInstance():
    def __init__(self, output_dir, huggingface_repo: str, keyword_dir: str, log_file_path:str, logger_id: str=datetime.now().strftime("%Y-%m-%d-%H:%M:%S")):
        self.log_file_path = log_file_path
        self.output_dir = output_dir
        self.huggingface_repo = huggingface_repo
        self._logger: logging.Logger | None = None
        self._keyword_loader: Keyword_loader | None = None
        self._dataset_loader: Data_loader | None = None
        self._data_writer: DF_writer | None = None
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
    def data_writer(self):
        if self._data_writer is not None:
            return self._data_writer
        data_writer = DF_writer(
            output_dir=self.output_dir
        )
        self._data_writer = data_writer
        return self._data_writer
    
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
        
    def _save_file_with_extension(self, df: pd.DataFrame, prefix: str, extension: str = "parquet"):
        if df is None or df.empty:
            self.logger.info("No data to save — DataFrame is empty.")
            return None

        save_method = {
            "parquet": self.data_writer.save_parquet,
            "csv": self.data_writer.save_csv,
            "xlsx": self.data_writer.save_excel
        }.get(extension.lower())

        if save_method is None:
            self.logger.warning(f"Unsupported file format: {extension}. Defaulting to parquet.")
            save_method = self.data_writer.save_parquet

        output_path = save_method(df, prefix)
        if output_path:
            self.logger.info(f"Saved output to: {output_path}")
        return output_path

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
            self._save_file_with_extension(matched_df, prefix="extracted_prs", extension="parquet")
        else:
            self.logger.info("No matching PRs found.")