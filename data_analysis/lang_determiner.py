import os
import pandas as pd
import fasttext
from tqdm import tqdm
from pydantic import BaseModel, ValidationError
import regex
from data_writer.df_writer import DF_writer
from langdetect import detect_langs, LangDetectException
import unicodedata

emoji_ranges = [
    (0x1F3FB, 0x1F3FF),
    (0x1F600, 0x1F64F),
    (0x1F900, 0x1F9FF),
    (0x1FA70, 0x1FAFF),
    (0x1F300, 0x1F5FF),
    (0x1F680, 0x1F6FF),
    (0x1F700, 0x1F77F),
    (0x1F780, 0x1F7FF),
    (0x1F800, 0x1F8FF),
]

emoji_singletons = {
    0x263A, 0x2639, 0x2763, 0x2764, 0x2600, 0x2601, 0x2602, 0x2603,
    0x2604, 0x260E, 0x2614, 0x2615, 0x2620, 0x2622, 0x2623, 0x2626,
    0x262A, 0x262E, 0x262F, 0x2638, 0x2640, 0x2642, 0x2660, 0x2663,
    0x2665, 0x2666, 0x2668, 0x267B, 0x267E, 0x267F, 0x2692, 0x2693,
    0x2695, 0x2696, 0x2697, 0x2699, 0x26A0, 0x26A1, 0x26AA, 0x26AB,
    0x26B0, 0x26B1, 0x26C4, 0x26C5, 0x26CE, 0x26D1, 0x26D3, 0x26EA,
    0x26F0, 0x26F1, 0x26F2, 0x26F3, 0x26F4, 0x26F5, 0x26F7, 0x26F8,
    0x26F9, 0x26FA, 0x2705, 0x2708, 0x2709, 0x270A, 0x270B, 0x270C,
    0x270D, 0x270F, 0x2712, 0x2714, 0x2716, 0x2721, 0x2728, 0x2733,
    0x2734, 0x2744, 0x2747, 0x274C, 0x274E, 0x2753, 0x2754, 0x2755,
    0x2757, 0x2795, 0x2796, 0x2797, 0x27A1, 0x27B0, 0x27BF, 0x2B05,
    0x2B06, 0x2B07, 0x2B1B, 0x2B1C, 0x2B50, 0x2B55,
    0x3030, 0x303D
}

def is_symbol(ch):
    return unicodedata.category(ch).startswith("S")

SKIP_CHARS = {
    "\u200b",
    "\u200c",
    "\u200d",
    "\uFE0F",
    "]",
    "[",
    "\u2192",
    "\u2139"
}

def is_emoji(ch: str) -> bool:
    cp = ord(ch)
    for start, end in emoji_ranges:
        if start <= cp <= end:
            return True
    return cp in emoji_singletons

def is_english_letter(ch: str) -> bool:
    return ("a" <= ch <= "z") or ("A" <= ch <= "Z")

symbols = set(r"""`~!@#$%^&*()_-+={[]}}|\:;"'<,>.?/""")
whitespace_pattern = regex.compile(r"\s")
number_pattern = regex.compile(r"[0-9]")

def count_non_english(text: str):
    ch_count = 0
    eng_letter_count = 0
    non_eng_letter_count = 0

    for ch in text:
        ch_count += 1

        if ch in SKIP_CHARS:
            continue

        if is_symbol(ch):
            continue

        if is_emoji(ch):
            continue

        if is_english_letter(ch):
            eng_letter_count += 1
            continue

        if ch in symbols:
            continue

        if whitespace_pattern.fullmatch(ch):
            continue

        if number_pattern.fullmatch(ch):
            continue
        non_eng_letter_count += 1

    return ch_count, eng_letter_count, non_eng_letter_count

class LangDeterminer:
    def __init__(self, logger, output_dir, df_writer: DF_writer):
        self.logger = logger
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.table_name = None
        self.data = None
        self.df_writer = df_writer

    def set_lang_determiner_data(self, data: pd.DataFrame, table_name):
        self.table_name = table_name.value if hasattr(table_name, "value") else table_name
        self.data = data
        self.logger.info(f"Set current data frame to: {self.table_name}")

    def rule_based_predict(self, text: str):
        if not isinstance(text, str) or not text.strip():
            return None

        ch_count, _, non_eng = count_non_english(text)

        if non_eng >= 1:
            return False
        return True
    # def rule_based_predict(self, text: str):
    #     if not isinstance(text, str) or not text.strip():
    #         return None

    #     text_norm = text.replace("\n", " ")

    #     labels, probs = self.ft.predict(text_norm)
    #     lang = labels[0].replace("__label__", "")
    #     prob = probs[0]

    #     if prob >= 0.95:
    #         return lang == "en"
    #     try:
    #         ld_lang = detect_langs(text_norm)
    #         if ld_lang:
    #             top = ld_lang[0]
    #             if top.prob >= 0.95:
    #                 return top.lang == "en"
    #     except LangDetectException:
    #         pass

    #     _, non_eng = count_non_english(text_norm)
    #     if non_eng > 5:
    #         return False
    #     return True

    def determine_lang(self):

        out_path = os.path.join(self.output_dir, "fast_rule_results.parquet")
        if os.path.exists(out_path):
            try:
                return pd.read_parquet(out_path)
            except Exception:
                print("[WARN] Existing parquet is corrupted. Recomputing...")
                os.remove(out_path)

        df = self.data
        if "id" not in df.columns:
            raise ValueError("DataFrame must have an 'id' column.")

        ids = df["id"].tolist()
        bodies = df["body"].tolist()

        if os.path.exists(out_path):
            print(f"[INFO] Loading existing file: {out_path}")
            return pd.read_parquet(out_path)

        results = []
        for pr_id, text in tqdm(zip(ids, bodies), total=len(ids), desc="Rule-based lang detection"):
            is_en = self.rule_based_predict(text)
            results.append({"id": pr_id, "is_en": is_en})

        final_df = pd.DataFrame(results)
        saved_path = self.df_writer.save_parquet(df=final_df, prefix="is_eng")
        print(f"[SAVED] {saved_path}")

        return final_df
