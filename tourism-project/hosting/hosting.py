from huggingface_hub import HfApi
import os

api = HfApi(token=os.getenv("HF_TOKEN"))
api.upload_folder(
    folder_path="/tourism_project/hosting",
    repo_id="Neethu-Sathyan/Tourism-project",
    repo_type="space"
)
