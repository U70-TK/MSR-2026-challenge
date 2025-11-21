import os
import pandas as pd
from datetime import datetime, timezone, timedelta

class DF_writer:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.utc_minus_4 = timezone(timedelta(hours=-5))

    def _get_timestamped_path(self, prefix: str, extension: str = "parquet") -> str:
        timestamp = datetime.now(self.utc_minus_4).strftime("%Y-%m-%d-%H:%M:%S")
        filename = f"{prefix}_{timestamp}.{extension}"
        return os.path.join(self.output_dir, filename)

    def save_parquet(self, df: pd.DataFrame, prefix: str = "output") -> str:
        if df is None or df.empty:
            return None
        output_path = self._get_timestamped_path(prefix, "parquet")
        df.to_parquet(output_path, index=False)
        return output_path

    def save_csv(self, df: pd.DataFrame, prefix: str = "output") -> str:
        if df is None or df.empty:
            return None
        output_path = self._get_timestamped_path(prefix, "csv")
        df.to_csv(output_path, index=False)
        return output_path

    def save_excel(self, df: pd.DataFrame, prefix: str = "output") -> str:
        if df is None or df.empty:
            return None
        output_path = self._get_timestamped_path(prefix, "xlsx")
        df.to_excel(output_path, index=False)
        return output_path
    
    def write_records_parquet(self, records: list[dict], prefix: str = "output") -> str:
        if not records:
            return None
        df = pd.DataFrame(records)
        return self.save_parquet(df, prefix)