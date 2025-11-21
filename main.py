from app_instance import AppInstance
from datetime import datetime, timedelta, timezone

utc_minus_4 = timezone(timedelta(hours=-5))
KEYWORD_DIR = "./keywords_regex"
LOG_DIR = f"./log/{datetime.now(utc_minus_4).strftime('%Y-%m-%d-%H:%M:%S')}"
HUGGING_FACE_REPO = "hao-li/AIDev/"
OUTPUT_DIR = "./output"
BASE_DATASET_URL = f"hf://datasets/{HUGGING_FACE_REPO}"

def main():
    
    app = AppInstance(
        output_dir=OUTPUT_DIR,
        keyword_dir=KEYWORD_DIR,
        log_file_path=LOG_DIR,
        huggingface_repo=BASE_DATASET_URL,
        logger_id=datetime.now(utc_minus_4).strftime('%Y-%m-%d-%H:%M:%S')
    )

    # app.match_llm_pr_desciption_title_merge()
    # app.detect_pr_description_lang()
    # app.match_human_pr_description_title_merge()
    app.determine_cwe_llm_pr()
    
if __name__ == '__main__':
    main()