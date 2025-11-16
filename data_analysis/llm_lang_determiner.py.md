```python
import os
import pandas as pd
import fasttext
from tqdm import tqdm
from ollama import chat
from pydantic import BaseModel, ValidationError
from typing import List
import re
from constants.dataset_info import AIDev
from langdetect import detect_langs, LangDetectException
import time

class LangOutput(BaseModel):
    is_english: bool

class LangOutput_Batch(BaseModel):
    is_english_lst: list[LangOutput]

FASTTEXT_MODEL = "./lid.176.bin"

class LLMLangDeterminer:
    def __init__(self, logger, output_dir):
        self.logger = logger
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.ft = fasttext.load_model(FASTTEXT_MODEL)
        self.table_name = None
        self.data = None
    
    def rule_based_predict(self, text: str):
        if not text or not isinstance(text, str):
            return None

        text_norm = text.replace("\n", " ")
        labels, probs = self.ft.predict(text_norm)
        lang = labels[0].replace("__label__", "")
        prob = probs[0]

        if prob >= 0.80:
            return lang == "en"

        try:
            ld_lang = detect_langs(text)
            if not ld_lang:
                return None
            top = ld_lang[0]
            if top.lang == "en" and top.prob >= 0.8:
                return True
            if top.lang != "en" and top.prob >= 0.8:
                return False
        except LangDetectException:
            pass

        return None
    
    def set_lang_determiner_data(self, data: pd.DataFrame, table_name: AIDev):
        if hasattr(table_name, "value"):
            table_name = table_name.value
        self.table_name = table_name
        self.data = data
        self.logger.info(f"Set current data frame to: {table_name}")

    def determine_lang(self):
        if self.table_name == AIDev.ALL_PULL_REQUEST.value:
            self._determine_lang_all_pull_request()
        else:
            if self.table_name:
                self.logger.info("Other tables are not supported right now. ")
            else:
                self.logger.info("Null data and table name. Please load data first. ")
            return None
    
    def llm_predict(self, texts: list[str], max_retries: int = 10):

        if len(texts) == 0:
            return []

        prompt = (
            "For each PR description below, determine whether it is written in English.\n"
            "Ignore code blocks, file names, and stack traces.\n"
            "You MUST respond with STRICT JSON only. "
            "No markdown, no comments, no explanations, no quotes.\n\n"
            "Schema:\n"
            f"{LangOutput_Batch.model_json_schema()}\n\n"
            "Return only the JSON object. Do not include backticks.\n"
            "The number of JSON objects MUST EQUAL the number of descriptions. "
            "Descriptions:\n"
        )

        for i, t in enumerate(texts):
            prompt += f"[{i}] {t}\n\n"

        def extract_json(text: str) -> str:
            m = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
            return m.group(1) if m else None

        for attempt in range(1, max_retries + 1):
            try:
                response = chat(
                    model="gpt-oss:120b",
                    messages=[{"role": "user", "content": prompt}],
                    format=LangOutput_Batch.model_json_schema(),
                )

                raw = response.message.content

                try:
                    batch = LangOutput_Batch.model_validate_json(raw)
                except ValidationError:
                    extracted = extract_json(raw)
                    if extracted:
                        batch = LangOutput_Batch.model_validate_json(extracted)
                    else:
                        raise

                if len(batch.is_english_lst) != len(texts):
                    print(
                        f"[WARN] LLM returned list of length {len(batch.is_english_lst)}, "
                        f"expected {len(texts)}. Retrying... (attempt {attempt})"
                    )
                    time.sleep(1.5)
                    continue
                return [x.is_english for x in batch.is_english_lst]

            except Exception as e:
                print(f"[WARN] LLM JSON parse failed (attempt {attempt}/{max_retries}). Error:")
                print(e)
                print("LLM output was:")
                print(raw)
                time.sleep(1.5)

        print("[ERROR] LLM failed after retries. Returning None for all entries.")
        return [None] * len(texts)

    def _determine_lang_all_pull_request(self, chunk_size=5000, batch_size=1):
        df = self.data
        if "id" not in df.columns:
            raise ValueError("DataFrame must contain an 'id' column for identification.")

        bodies = df["body"].tolist()
        ids = df["id"].tolist()
        n = len(bodies)

        cp_dir = os.path.join(self.output_dir, "checkpoints")
        os.makedirs(cp_dir, exist_ok=True)

        fast_rule_file = os.path.join(cp_dir, "fast_rule_results.parquet")

        if os.path.exists(fast_rule_file):
            print(f"Loading existing fast rule results from: {fast_rule_file}")
            fast_df = pd.read_parquet(fast_rule_file)
            merged = df[["id"]].merge(fast_df, on="id", how="left")
            return merged["is_en"]

        results = []
        for idx, (pr_id, text) in tqdm(
            list(enumerate(zip(ids, bodies))),
            total=n,
            desc="FastText + langdetect pass",
        ):
            res = self.rule_based_predict(text)
            results.append({"id": pr_id, "is_en": res})

        fast_df = pd.DataFrame(results)
        fast_df.to_parquet(fast_rule_file, index=False)
        print(f"Saved fast rule results to: {fast_rule_file}")

        merged = df[["id"]].merge(fast_df, on="id", how="left")
        return merged["is_en"]

        # df = self.data
        # bodies = df["body"].tolist()
        # n = len(bodies)

        # cp_dir = os.path.join(self.output_dir, "checkpoints")
        # os.makedirs(cp_dir, exist_ok=True)

        # checkpoint_file = os.path.join(cp_dir, "langs_checkpoint.parquet")

        # if os.path.exists(checkpoint_file):
        #     print("Loading checkpoint...")
        #     series = pd.read_parquet(checkpoint_file)["is_en"].tolist()
        #     print(f"Loaded checkpoint: {sum(x is not None for x in series)} completed, {n - sum(x is not None for x in series)} pending.")
        # else:
        #     series = [None] * n

        # langs = series
        # uncertain_bodies = []
        # uncertain_indices = []

        # for idx, text in tqdm(enumerate(bodies), total=n, desc="FastText pass"):
        #     if langs[idx] is not None:
        #         continue

        #     res = self.rule_based_predict(text)

        #     if res is not None:
        #         langs[idx] = res

        #     else:
        #         uncertain_bodies.append(text)
        #         uncertain_indices.append(idx)

        #     if idx % 50000 == 0 and idx > 0:
        #         pd.Series(langs, name="is_en").to_frame().to_parquet(checkpoint_file)
        #         print(f"Checkpoint saved at FastText index {idx}.")

        # for start in tqdm(range(0, len(uncertain_bodies), batch_size), desc="LLM batches"):
        #     batch = uncertain_bodies[start:start + batch_size]
        #     results = self.llm_predict(batch)

        #     for j, result in enumerate(results):
        #         original = uncertain_indices[start + j]
        #         langs[original] = result

        #     pd.Series(langs, name="is_en").to_frame().to_parquet(checkpoint_file)
        #     print(f"Checkpoint saved after LLM batch {start//batch_size}.")

        # final_series = pd.Series(langs, name="is_en")
        # for c in range(0, n, chunk_size):
        #     chunk = final_series[c:c + chunk_size]
        #     path = os.path.join(cp_dir, f"chunk_{c // chunk_size:03d}.parquet")
        #     chunk.to_frame().to_parquet(path)

        # return final_series```