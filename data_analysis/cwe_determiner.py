import os
import json
import logging
from typing import List, Dict, Any, Optional
from ollama._types import ResponseError
import pandas as pd
import regex
from tqdm import tqdm
from transformers import AutoTokenizer
from ollama import chat, ChatResponse
from pydantic import BaseModel, ValidationError

from data_writer.df_writer import DF_writer
from constants.dataset_info import AIDev

class CWEItem(BaseModel):
    cwe_id: int
    cwe_title: str

class ResponseSchema(BaseModel):
    is_security_patch: bool
    cwe_lst: List[CWEItem]

class CWEDeterminer:

    MAX_INPUT_TOKENS = 30000
    MAX_TOTAL_TOKENS = 40000
    MAX_COMPLETION_TOKENS = 1000

    def __init__(
        self,
        table_name: AIDev | str,
        df: pd.DataFrame,
        logger: logging.Logger,
        data_writer: DF_writer,
        cwe_catalog_path: str = "./data/cwe_children_1000.parquet",
    ):
        self.logger = logger
        self.data_writer = data_writer
        self.df = df
        self.calculated = 0

        if hasattr(table_name, "value"):
            table_name = table_name.value
        self.table_name = table_name

        self.tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-120b")

        self.cwe_catalog = pd.read_parquet(cwe_catalog_path)
        self.cwe_catalog["_title_lower"] = self.cwe_catalog["cwe_title"].str.lower()

        self._current_full_diff: str = ""

    def _lookup_cwe_impl(self, title: str) -> Optional[Dict[str, Any]]:

        if not title:
            return None

        q = title.lower().strip()
        if not q:
            return None

        mask = self.cwe_catalog["_title_lower"].str.contains(q, na=False)
        df = self.cwe_catalog[mask]

        if df.empty:
            mask2 = self.cwe_catalog["_title_lower"].apply(
                lambda t: isinstance(t, str) and t in q
            )
            df = self.cwe_catalog[mask2]

        if df.empty:
            import difflib

            candidates = []
            for _, row in self.cwe_catalog.iterrows():
                t = row.get("_title_lower")
                if not isinstance(t, str):
                    continue
                ratio = difflib.SequenceMatcher(None, q, t).ratio()
                candidates.append((ratio, row))

            candidates.sort(key=lambda x: x[0], reverse=True)
            if candidates and candidates[0][0] > 0.60:
                df = pd.DataFrame([candidates[0][1]])

        if df.empty:
            return None

        row = df.iloc[0]
        return {
            "cwe_id": int(row["cwe_id"]),
            "cwe_title": row["cwe_title"],
            "cwe_description": row["cwe_description"],
        }

    def _lookup_cwe_text(self, title: str) -> str:

        result = self._lookup_cwe_impl(title)
        if not result:
            return (
                f"No matching CWE found for title: {title!r}. "
                "Please reconsider or try a more precise CWE title."
            )

        return (
            f"CWE-{result['cwe_id']}: {result['cwe_title']}\n"
            f"Description: {result['cwe_description']}"
        )

    def _extract_file_chunks_fuzzy(self, full_diff: str, target_filename: str) -> str:

        if not full_diff or not target_filename:
            return ""

        target_norm = target_filename.strip()
        target_norm_lower = target_norm.lower()

        lines = full_diff.splitlines()
        n = len(lines)

        headers = []
        for i, line in enumerate(lines):
            m = regex.match(r"^=== CHUNK (.+?) ===\s*$", line)
            if m:
                header_text = m.group(1).strip()
                headers.append((i, header_text))

        if not headers:
            return ""

        chunk_ranges = []
        for idx, (start_idx, header_text) in enumerate(headers):
            end_idx = headers[idx + 1][0] - 1 if idx + 1 < len(headers) else n - 1
            chunk_ranges.append((start_idx, end_idx, header_text))

        best_range: Optional[tuple[int, int]] = None

        for (s, e, h) in chunk_ranges:
            if h.strip() == target_norm:
                best_range = (s, e)
                break

        if best_range is None:
            for (s, e, h) in chunk_ranges:
                h_lower = h.strip().lower()
                if target_norm_lower in h_lower or h_lower in target_norm_lower:
                    best_range = (s, e)
                    break

        if best_range is None:
            import os as _os

            base = _os.path.basename(target_norm_lower)
            for (s, e, h) in chunk_ranges:
                h_lower = h.strip().lower()
                if base and base in h_lower:
                    best_range = (s, e)
                    break

        if best_range is None:
            return ""

        s_idx, e_idx = best_range
        chunk_lines = lines[s_idx : e_idx + 1]
        return "\n".join(chunk_lines) + "\n"

    def _get_file_diff_impl(self, filename: str) -> str:
        return self._extract_file_chunks_fuzzy(self._current_full_diff, filename)

    def _extract_json_fallback(self, s: str) -> Optional[Dict[str, Any]]:
        if not s or not isinstance(s, str):
            return None

        bool_pattern = regex.compile(
            r'is_security_patch"?\s*[:=]\s*'
            r'(true|false|"true"|"false"|True|False)',
            flags=regex.IGNORECASE,
        )
        m_bool = bool_pattern.search(s)
        if m_bool:
            raw = m_bool.group(1).strip('"').lower()
            is_sec = raw == "true"
        else:
            is_sec = False

        cwe_results = []

        dict_brace_pattern = regex.compile(
            r'\{\s*CWE[-:]?(?P<id>\d+)\s*[:=]\s*(?P<title>[^}]+?)\s*\}',
            flags=regex.IGNORECASE,
        )
        for m in dict_brace_pattern.finditer(s):
            cwe_results.append(
                {
                    "cwe_id": int(m.group("id")),
                    "cwe_title": m.group("title").strip().strip('"').strip(),
                }
            )

        dict_style_pattern = regex.compile(
            r'"?CWE[-:]?(?P<id>\d+)"?\s*[:=]\s*"?(?P<title>[^",}\]\r\n]+)"?',
            flags=regex.IGNORECASE,
        )
        for m in dict_style_pattern.finditer(s):
            cwe_results.append(
                {
                    "cwe_id": int(m.group("id")),
                    "cwe_title": m.group("title").strip(),
                }
            )

        cwe_loose_pattern = regex.compile(
            r"""
            (?P<obj>
                \{
                    [^{}]*
                \}
                |
                CWE[-]?(?P<id1>\d+)\s+(?P<title1>[A-Za-z0-9 ,._\-()]+)
                |
                (?P<id2>\d+)\s+(?P<title2>[A-Za-z0-9 ,._\-()]+)
            )
            """,
            flags=regex.VERBOSE | regex.IGNORECASE,
        )

        for m in cwe_loose_pattern.finditer(s):
            obj = m.group("obj")
            if obj and obj.startswith("{"):
                inner_id = regex.search(
                    r'cwe_id"?\s*[:=]\s*(\d+)', obj, flags=regex.IGNORECASE
                )
                inner_title = regex.search(
                    r'cwe_title"?\s*[:=]\s*"([^"]+)"', obj, flags=regex.IGNORECASE
                )
                if inner_id and inner_title:
                    cwe_results.append(
                        {
                            "cwe_id": int(inner_id.group(1)),
                            "cwe_title": inner_title.group(1).strip(),
                        }
                    )
                continue

            if m.group("id1") and m.group("title1"):
                cwe_results.append(
                    {
                        "cwe_id": int(m.group("id1")),
                        "cwe_title": m.group("title1").strip(),
                    }
                )
                continue

            if m.group("id2") and m.group("title2"):
                cwe_results.append(
                    {
                        "cwe_id": int(m.group("id2")),
                        "cwe_title": m.group("title2").strip(),
                    }
                )
                continue

        uniq: Dict[int, Dict[str, Any]] = {}
        for c in cwe_results:
            cid = c["cwe_id"]
            title = c["cwe_title"].strip()
            if not title or cid is None:
                continue
            uniq[cid] = {"cwe_id": cid, "cwe_title": title}

        return {
            "is_security_patch": is_sec,
            "cwe_lst": list(uniq.values()),
        }

    def _exceeds_token_limits(self, body: str, code_diff_full: str) -> bool:
        text = (body or "") + (code_diff_full or "")
        tokens = self.tokenizer.encode(text)
        n_tokens = len(tokens)

        if n_tokens >= self.MAX_INPUT_TOKENS:
            # self.logger.info(
            #     f"Skipping PR due to input tokens = {n_tokens} "
            #     f"(>= {self.MAX_INPUT_TOKENS})"
            # )
            return True

        if n_tokens + self.MAX_COMPLETION_TOKENS > self.MAX_TOTAL_TOKENS:
            # self.logger.info(
            #     f"Skipping PR due to total tokens {n_tokens} + "
            #     f"{self.MAX_COMPLETION_TOKENS} > {self.MAX_TOTAL_TOKENS}"
            # )
            return True

        return False

    def _llm_security_classify_with_tools(
        self,
        body: str,
        files_changed: List[str],
        code_diff_full: str,
    ) -> Optional[ResponseSchema]:
        if self._exceeds_token_limits(body, code_diff_full):
            return None

        self._current_full_diff = code_diff_full or ""

        def lookup_cwe(title: str) -> str:
            return self._lookup_cwe_text(title)

        def get_file_diff(filename: str) -> str:
            return self._get_file_diff_impl(filename)

        available_functions = {
            "lookup_cwe": lookup_cwe,
            "get_file_diff": get_file_diff,
        }

        files_block = "\n".join(f"- {path}" for path in files_changed)

        user_content = (
            "You are a professional software developer and security engineer.\n"
            "You will classify whether this pull request is a SECURITY PATCH and\n"
            "determine the correct CWE categories.\n\n"
            "VERY IMPORTANT:\n"
            "1. First, examine the PR description and list of changed files.\n"
            "2. Then choose EXACTLY ONE file that is most relevant for deciding\n"
            "   whether this PR fixes a security issue.\n"
            "3. Call get_file_diff(filename=...) ONCE using a filename from the list.\n"
            "4. Carefully read the returned diff to understand the actual code change.\n\n"
            "5. If you suspect this *is* a security patch:\n"
            "   - Think of several possible CWE titles that might apply.\n"
            "   - For EACH suspected CWE, call lookup_cwe(title=...).\n"
            "   - Read the CWE description returned by the tool.\n"
            "   - ONLY keep CWEs whose description truly matches the PR context.\n"
            "   - Discard any CWE that is only loosely related.\n\n"
            "6. Once you have fully used the tools and made your decision, respond\n"
            "   ONLY with JSON matching this schema (no markdown, no code fences):\n"
            f"{ResponseSchema.model_json_schema()}\n"
            "Do NOT include explanations, comments, or extra text.\n\n"
            "### Pull Request Description\n"
            f"{body}\n\n"
            "### Files Changed (you MUST choose one of these for get_file_diff)\n"
            f"{files_block}\n"
        )

        messages: List[Dict[str, Any]] = [
            {
                "role": "user",
                "content": user_content,
            }
        ]

        while True:
            max_retries = 3
            for attempt in range(1, max_retries + 1):
                try:
                    response: ChatResponse = chat(
                        model="gpt-oss:120b",
                        messages=messages,
                        tools=list(available_functions.values()),
                        think=True,
                        options={"num_predict": self.MAX_COMPLETION_TOKENS},
                    )
                    break

                except ResponseError as e:
                    raw = getattr(e, "message", "")
                    self.logger.error(
                        f"[Retry {attempt}/{max_retries}] Ollama parse error: {raw[:200]}"
                    )

                    if attempt == max_retries:
                        self.logger.error(
                            f"[Abort] Ollama still failing after {max_retries} retries → skipping this PR."
                        )
                        return None

                    messages.append({
                        "role": "user",
                        "content":
                            "Your previous output was not valid JSON or a valid tool call. "
                            "Please strictly follow the JSON format or produce a valid tool call."
                    })
                    continue

            # if response.message.content:
                # self.logger.debug(
                #     f"Assistant content (partial/final): {response.message.content[:500]}"
                # )

            messages.append(response.message)

            tool_calls = response.message.tool_calls or []

            if tool_calls:
                for tc in tool_calls:
                    fn_name = tc.function.name
                    fn_args = tc.function.arguments or {}

                    if fn_name not in available_functions:
                        self.logger.warning(
                            f"[Tool Error] Unknown tool call '{fn_name}'. "
                            f"Args: {fn_args}. Skipping this PR safely."
                        )
                        return None

                    try:
                        result = available_functions[fn_name](**fn_args)
                    except Exception as e:
                        self.logger.error(
                            f"[Tool Error] Tool '{fn_name}' failed with args {fn_args}: {e!r}"
                        )
                        return None

                    messages.append(
                        {
                            "role": "tool",
                            "tool_name": fn_name,
                            "content": str(result),
                        }
                    )

                continue


            break

        raw = response.message.content or ""
        if not raw.strip():
            self.logger.warning("Final assistant content empty after tool loop.")
            return None

        self.logger.info(f"Final raw model output (truncated): {raw[:500]}")

        try:
            resp = ResponseSchema.model_validate_json(raw)
            if not resp.is_security_patch:
                resp.cwe_lst = []
            return resp
        except ValidationError:
            self.logger.warning("JSON validation failed, attempting regex fallback.")
            fallback = self._extract_json_fallback(raw)
            if not fallback:
                return None

            try:
                resp = ResponseSchema(**fallback)
                if not resp.is_security_patch:
                    resp.cwe_lst = []
                return resp
            except ValidationError:
                self.logger.error("Fallback JSON still invalid for ResponseSchema.")
                return None

    def _normalize_files_changed(self, value) -> List[str]:
        if value is None:
            return []

        if isinstance(value, list):
            return [str(v).strip() for v in value if str(v).strip()]

        try:
            import numpy as np

            if isinstance(value, np.ndarray):
                return [str(v).strip() for v in value.tolist() if str(v).strip()]
        except ImportError:
            pass

        value_str = str(value).strip()
        if not value_str:
            return []

        if value_str.startswith("[") and value_str.endswith("]"):
            try:
                parsed = json.loads(value_str.replace("'", '"'))
                if isinstance(parsed, list):
                    return [str(v).strip() for v in parsed if str(v).strip()]
            except Exception:
                pass
            try:
                parsed = eval(value_str)
                if isinstance(parsed, list):
                    return [str(v).strip() for v in parsed if str(v).strip()]
            except Exception:
                pass

        if "," in value_str:
            parts = [p.strip() for p in value_str.split(",") if p.strip()]
            if parts:
                return parts

        if "/" in value_str or "." in value_str:
            return [value_str]

        return []

    def run_llm_for_dataframe(
        self,
        chunk_size: int = 100,
        prefix: str = "cwe_output",
    ) -> None:

        for col in ("body", "code_diff", "files_changed"):
            if col not in self.df.columns:
                raise ValueError(f"DataFrame missing required column: {col}")

        results: List[Dict[str, Any]] = []
        total = len(self.df)

        self.logger.info(f"Starting CWE classification on {total} rows.")

        for idx, row in tqdm(self.df.iterrows(), total=total, desc="Classifying PRs"):
            pr_id = row.get("id", idx)
            body = row.get("body", "") or ""
            code_diff_full = row.get("code_diff", "") or ""

            files_changed_raw = row.get("files_changed")
            files_changed = self._normalize_files_changed(files_changed_raw)

            if not files_changed:
                self.logger.info(f"[PR {pr_id}] No files_changed, skipping.")
                continue

            resp = self._llm_security_classify_with_tools(
                body=body,
                files_changed=files_changed,
                code_diff_full=code_diff_full,
            )

            if resp is None:
                self.logger.info("resp is None, returning False")
                results.append(
                    {
                        "id": pr_id,
                        "is_security_patch": False,
                        "cwe_lst": [],
                    }
                )
            else:
                self.calculated += 1
                results.append(
                    {
                        "id": pr_id,
                        "is_security_patch": resp.is_security_patch,
                        "cwe_lst": [c.model_dump() for c in resp.cwe_lst],
                    }
                )
            self.logger.info(f"Non-skipped ones: {self.calculated}")
            # self.logger.info(f"Current length of results: {len(results)}")

            if len(results) >= chunk_size:
                self.logger.info(f"Writing Parquet chunk at dataframe index {idx}")
                self.data_writer.write_records_parquet(
                    records=results,
                    prefix=f"{self.table_name}_{prefix}",
                )
                results.clear()

        if results:
            self.logger.info("Writing final Parquet chunk.")
            self.data_writer.write_records_parquet(
                records=results,
                prefix=f"{self.table_name}_{prefix}_final",
            )

        self.logger.info("CWE classification completed.")