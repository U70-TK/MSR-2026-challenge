import os
from ollama import chat
from tqdm import tqdm
from pydantic import BaseModel, ValidationError
from data_writer.df_writer import DF_writer
from constants.dataset_info import AIDev
import logging
import pandas as pd
import json
import regex

class CWE_Response(BaseModel):
    cwe_id: int
    cwe_title: str

class Response_Structure(BaseModel):
    is_security_patch: bool
    cwe_lst: list[CWE_Response]

class CWEDeterminer():
    def __init__(self, table_name: AIDev | str, df: pd.DataFrame, logger: logging.Logger, data_writer: DF_writer):
        self.logger = logger
        self.data_writer = data_writer
        self.df = df
        self.table_name = table_name
        if hasattr(table_name, "value"):
            table_name = table_name.value

    def _extract_json(self, s: str):
        if not s:
            return None

        bool_pattern = regex.compile(
            r'is_security_patch"?\s*[:=]\s*(true|false|"true"|"false"|True|False)',
            flags=regex.IGNORECASE
        )
        bool_match = bool_pattern.search(s)

        if bool_match:
            raw_bool = bool_match.group(1).strip('"').lower()
            is_sec = (raw_bool == "true")
        else:
            is_sec = False

        cwe_results = []

        dict_brace_pattern = regex.compile(
            r'\{\s*CWE[-:]?(?P<id>\d+)\s*[:=]\s*(?P<title>[^}]+?)\s*\}',
            flags=regex.IGNORECASE
        )
        for m in dict_brace_pattern.finditer(s):
            cwe_results.append({
                "cwe_id": int(m.group("id")),
                "cwe_title": m.group("title").strip().strip('"').strip()
            })

        dict_style_pattern = regex.compile(
            r'"?CWE[-:]?(?P<id>\d+)"?\s*[:=]\s*"?(?P<title>[^",}\]\r\n]+)"?',
            flags=regex.IGNORECASE
        )
        for m in dict_style_pattern.finditer(s):
            cwe_results.append({
                "cwe_id": int(m.group("id")),
                "cwe_title": m.group("title").strip()
            })

        cwe_pattern = regex.compile(
            r'''
            (?P<obj>
                \{
                    (?:
                        [^{}]*
                    )
                \}
                |
                (?:
                    CWE[-]?(?P<id1>\d+)\s+(?P<title1>[A-Za-z0-9 ,._\-()]+)
                )
                |
                (?:
                    (?P<id2>\d+)\s+(?P<title2>[A-Za-z0-9 ,._\-()]+)
                )
            )
            ''',
            flags=regex.VERBOSE | regex.IGNORECASE
        )

        for m in cwe_pattern.finditer(s):
            block = m.group("obj")

            if block and block.startswith("{"):
                id_match = regex.search(r'cwe_id"?\s*[:=]\s*(\d+)', block, flags=regex.IGNORECASE)
                title_match = regex.search(r'cwe_title"?\s*[:=]\s*"([^"]+)"', block, flags=regex.IGNORECASE)
                if id_match and title_match:
                    cwe_results.append({
                        "cwe_id": int(id_match.group(1)),
                        "cwe_title": title_match.group(1).strip()
                    })
                continue

            if m.group("id1") and m.group("title1"):
                cwe_results.append({
                    "cwe_id": int(m.group("id1")),
                    "cwe_title": m.group("title1").strip()
                })
                continue

            if m.group("id2") and m.group("title2"):
                cwe_results.append({
                    "cwe_id": int(m.group("id2")),
                    "cwe_title": m.group("title2").strip()
                })
                continue
        uniq = {c["cwe_id"]: c for c in cwe_results}

        return {
            "is_security_patch": is_sec,
            "cwe_lst": list(uniq.values())
        }
        
    def _llm_determine(self, code_diff: str, text: str, max_retry: int = 10):
        if not text:
            return None
        
        prompt = (
            "You are a professional software developer and system security expert.\n"
            "Decide if the following description and code diffs of a pull request "
            "is likely a security patch.\n"
            "Also decide the related list of Common Weakness Enumeration.\n"
            "No markdown, no comments, no explanations, no quotes.\n\n"
            "Schema:\n"
            f"{Response_Structure.model_json_schema()}\n\n"
            "Description:\n"
            f"{text}\n\n"
            "Code diffs:\n"
            f"{code_diff}\n\n"
            "Result:"
        )

        for attempt in range(1, max_retry + 1):
            try:
                response = chat(
                    model="gpt-oss:120b",
                    messages=[{"role": "user", "content": prompt}],
                    format=Response_Structure.model_json_schema(),
                )

                raw = response.message.content

                try:
                    response_verified = Response_Structure.model_validate_json(raw)
                    return response_verified

                except ValidationError:
                    self.logger.warning(
                        f"[Attempt {attempt}] JSON validation failed, attempting regex parse."
                    )
                    extracted_dict = self._extract_json(raw)

                    if extracted_dict:
                        try:
                            response_verified = Response_Structure(**extracted_dict)
                            return response_verified
                        except ValidationError:
                            self.logger.error(
                                f"[Attempt {attempt}] Regex extracted JSON but still invalid schema."
                            )

            except Exception as e:
                self.logger.error(f"[Attempt {attempt}] LLM call failure: {repr(e)}")

        self.logger.error("Max retries reached. Returning None.")
        return None

    def run_llm_for_dataframe(self, chunk_size: int = 100, prefix: str = "cwe_output"):
        if "body" not in self.df or "code_diff" not in self.df:
            raise ValueError("DataFrame must contain 'body' and 'code_diff' columns.")

        results = []
        total_rows = len(self.df)

        self.logger.info(f"Starting CWE LLM classification on {total_rows} rows.")

        for idx, row in tqdm(self.df.iterrows(), total=total_rows, desc="Processing PRs"):
            pr_id = row.get("id", idx)

            code_diff = row.get("code_diff", "")
            text = row.get("body", "")

            result_obj = self._llm_determine(code_diff, text)

            if result_obj:
                results.append({
                    "id": pr_id,
                    "is_security_patch": result_obj.is_security_patch,
                    "cwe_lst": [
                        {"cwe_id": c.cwe_id, "cwe_title": c.cwe_title}
                        for c in result_obj.cwe_lst
                    ]
                })
            else:
                results.append({
                    "id": pr_id,
                    "is_security_patch": False,
                    "cwe_lst": []
                })

            if len(results) >= chunk_size:
                self.logger.info(f"Saving parquet chunk at row {idx}")
                self.data_writer.write_records_parquet(
                    records=results,
                    prefix=f"{self.table_name}_chunk"
                )
                results.clear()

        if results:
            self.logger.info("Saving final parquet chunk.")
            self.data_writer.write_records_parquet(
                records=results,
                prefix=f"{self.table_name}_final"
            )

        self.logger.info("CWE classification completed.")