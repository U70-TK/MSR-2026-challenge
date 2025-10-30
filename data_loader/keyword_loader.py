import re
from tqdm import tqdm
import os

class Keyword_loader():
    def __init__(self, keyword_dir_path: str, logger, extension:str=".txt"):
        self.keyword_dir_path = keyword_dir_path
        self.extention = extension
        self.keyword_regex_lst = []
        self.file_path_lst = []
        self.compiled_regex_lst = []
        self._find_all_files()
        self.logger = logger

    def _find_all_files(self):
        for root, _, files in os.walk(self.keyword_dir_path):
            for filename in files:
                if filename.endswith(self.extention):
                    self.file_path_lst.append(os.path.join(root,filename))

    def _get_all_files(self):
        return self.file_path_lst
    
    def _get_all_regex(self):
        return self.keyword_regex_lst
    
    def _load_keywords(self):
        self.logger.info(f"{'-'*10} Loading keyword list {'-'*10}")
        progress_bar = tqdm(total=len(self.file_path_lst))
        for file_path in self.file_path_lst:
            with open(file_path, 'r') as file:
                for line in file:
                    if line.strip().startswith("->"):
                        continue
                    if line and line not in self.keyword_regex_lst:
                        self.keyword_regex_lst.append(line)
            progress_bar.update(1)
        self.logger.info(f"{'-'*10} Loading keyword finished {'-'*10}")

    def _compile_regex(self):
        for regex_str in self.keyword_regex_lst:
            try:
                pattern = re.compile(regex_str.strip(), re.IGNORECASE)
                self.compiled_regex_lst.append(pattern)
            except re.error as e:
                self.logger.info(f"Skipped invalid regex: {regex_str.strip()} ({e})")
            
    def match_any(self, text:str) -> bool:
        if not self.keyword_regex_lst:
            self._load_keywords()
            if not self.keyword_regex_lst:
                self.logger.warning("Error: Keyword list is empty")
                return False
            
        if self.keyword_regex_lst and not self.compiled_regex_lst:
            self._compile_regex()
            if not self.compiled_regex_lst:
                self.logger.warning("Error: Compiled keyword list is empty")
                return False
        
        for pattern in self.compiled_regex_lst:
            if pattern.search(text):
                return True
        return False