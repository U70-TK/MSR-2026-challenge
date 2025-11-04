import logging
from data_loader.keyword_loader import Keyword_loader
from data_loader.dataset_loader import Data_loader
from data_writer.df_writer import DF_writer
from datetime import datetime
import os
import pandas as pd
from tqdm import tqdm
from constants.dataset_info import AIDev
from multiprocessing import Pool, cpu_count, current_process
from collections import Counter
from typing import List

def _process_row(args):
    row, compiled_regex_lst = args
    body = str(row["body"]) if pd.notna(row["body"]) else ""
    local_counts = Counter()
    matched_id = None

    for pattern, regex_str in compiled_regex_lst:
        if pattern.search(body):
            matched_id = row["id"]
            local_counts[regex_str] += 1

    return (matched_id, local_counts)

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
            self.logger.warning(f"Unsupported format: {extension}, exporting parquet.")
            save_method = self.data_writer.save_parquet

        output_path = save_method(df, prefix)
        if output_path:
            self.logger.info(f"Saved output to: {output_path}")
        return output_path
    
    def match_dataframe_return_id(self, matching_func: str, subset_name, matching_column: str) -> List:
        if hasattr(subset_name, "value"):
            subset_name = subset_name.value

        matching_df = self._load_table_with_name(subset_name)

        if matching_df is None or matching_column not in matching_df:
            self.logger.warning(f"Dataframe {matching_df} missing {matching_column} column. ")
            return
        
        total_rows = len(matching_df)
        self.logger.info(f"Scanning {total_rows:,} {subset_name} dynamically...")

        if not self.keyword_loader.compiled_regex_lst:
            self.keyword_loader._load_keywords()
        compiled_regex_lst = self.keyword_loader.compiled_regex_lst
        num_cpus = max(1, cpu_count() - 1)

        with Pool(processes=num_cpus) as pool:
            results = list(
                tqdm(
                    pool.imap_unordered(
                        exec(f"{matching_func}()"),
                        ((row, compiled_regex_lst) for _, row in matching_df.iterrows()),
                        chunksize=100
                    ),
                    total=total_rows,
                    desc="Multi-threaded",
                    dynamic_ncols=True
                )
            )

        all_counts = Counter()
        matched_ids = []
        for res_id, local_counts in tqdm(results, total=total_rows):
            if res_id:
                matched_ids.append(res_id)
            all_counts.update(local_counts)

        for regex, count in all_counts.items():
            self.keyword_loader.regex_match_counts[regex] += count
            
        self.keyword_loader.log_regex_statistics()

        return matched_ids
    
    def match_pr_commits(self):
        matched_id = self.match_dataframe_return_id(
            matching_func=_process_row.__name__
            subset_name=AIDev.PR_COMMITS
            
        )

        matched_ids_df = pd.DataFrame(
            matched_ids,
            columns=["matched_ids"]
        )

        self._save_file_with_extension(
            matched_ids_df, 
            f"human_pr_description",
            extension="parquet"
        )

    def match_human_pr_description(self):
        human_pr = self._load_table_with_name(AIDev.HUMAN_PULL_REQUEST)
        if human_pr is None or "body" not in human_pr.columns:
            self.logger.warning("Missing 'body' column.")
            return
        
        total_rows = len(human_pr)
        self.logger.info(f"Scanning {total_rows:,} Human PR descriptions dynamically...")

        if not self.keyword_loader.compiled_regex_lst:
            self.keyword_loader._load_keywords()
        compiled_regex_lst = self.keyword_loader.compiled_regex_lst
        num_cpus = max(1, cpu_count() - 1)

        with Pool(processes=num_cpus) as pool:
            results = list(
                tqdm(
                    pool.imap_unordered(
                        _process_row,
                        ((row, compiled_regex_lst) for _, row in human_pr.iterrows()),
                        chunksize=100
                    ),
                    total=total_rows,
                    desc="Multi-threaded",
                    dynamic_ncols=True
                )
            )

        all_counts = Counter()
        matched_ids = []
        for res_id, local_counts in tqdm(results, total=total_rows):
            if res_id:
                matched_ids.append(res_id)
            all_counts.update(local_counts)

        for regex, count in all_counts.items():
            self.keyword_loader.regex_match_counts[regex] += count
            
        self.keyword_loader.log_regex_statistics()

        matched_ids_df = pd.DataFrame(
            matched_ids,
            columns=["matched_ids"]
        )

        self._save_file_with_extension(
            matched_ids_df, 
            f"human_pr_description",
            extension="parquet"
        )

    def match_pr_description(self):
        all_pr_request = self._load_table_with_name(AIDev.ALL_PULL_REQUEST)
        if all_pr_request is None or "body" not in all_pr_request.columns:
            self.logger.warning("Missing 'body' column.")
            return

        total_rows = len(all_pr_request)
        self.logger.info(f"Scanning {total_rows:,} PR descriptions dynamically...")

        if not self.keyword_loader.compiled_regex_lst:
            self.keyword_loader._load_keywords()
        compiled_regex_lst = self.keyword_loader.compiled_regex_lst

        num_cpus = max(1, cpu_count() - 1)

        with Pool(processes=num_cpus) as pool:
            results = list(
                tqdm(
                    pool.imap_unordered(
                        _process_row,
                        ((row, compiled_regex_lst) for _, row in all_pr_request.iterrows()),
                        chunksize=100
                    ),
                    total=total_rows,
                    desc="Multi-threaded",
                    dynamic_ncols=True
                )
            )

        all_counts = Counter()
        matched_ids = []
        for res_id, local_counts in tqdm(results, total=total_rows):
            if res_id:
                matched_ids.append(res_id)
            all_counts.update(local_counts)

        for regex, count in all_counts.items():
            self.keyword_loader.regex_match_counts[regex] += count
            
        self.keyword_loader.log_regex_statistics()

        matched_ids_df = pd.DataFrame(
            matched_ids,
            columns=["matched_ids"]
        )

        self._save_file_with_extension(
            matched_ids_df, 
            f"pr_description",
            extension="parquet"
        )