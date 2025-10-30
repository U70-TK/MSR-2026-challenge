from app_instance import AppInstance
from datetime import datetime

KEYWORD_DIR = "./keywords_regex"
LOG_DIR = f"./log/{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}"
HUGGING_FACE_REPO = "hf://datasets/hao-li/AIDev/"

def main():
    
    app = AppInstance(
        keyword_dir=KEYWORD_DIR,
        log_file_path=LOG_DIR,
        huggingface_repo=HUGGING_FACE_REPO
    )

    app.match_pr_description()
    

if __name__ == '__main__':
    main()