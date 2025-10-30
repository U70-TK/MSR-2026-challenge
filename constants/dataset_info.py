from enum import Enum

class AIDev(Enum):
    ALL_PULL_REQUEST = "all_pull_request"
    ALL_REPOSITORY = "all_repository"
    ALL_USER = "all_user"
    HUMAN_PR_TASK_TYPE = "human_pr_task_type"
    HUMAN_PULL_REQUEST = "human_pull_request"
    ISSUE = "issue"
    PR_COMMENTS = "pr_comments"
    PR_COMMIT_DETAILS = "pr_commit_details"
    PR_COMMITS = "pr_commits"
    PR_REVIEW_COMMENTS = "pr_review_comments"
    PR_REVIEW_COMMENTS_V2 = "pr_review_comments_v2"
    PR_REVIEWS = "pr_reviews"
    PR_TASK_TYPE = "pr_task_type"
    PR_TIMELINE = "pr_timeline"
    PULL_REQUEST = "pull_request"
    RELATED_ISSUE = "related_issue"
    REPOSITORY = "repository"
    USER = "user"

    @classmethod
    def all_parquet_files(cls):
        return {item.value : f"{item.value}.parquet" for item in cls}