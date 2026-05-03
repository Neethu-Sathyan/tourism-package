from huggingface_hub import HfApi
import os

api = HfApi(token=os.getenv("HF_TOKEN"))
api.upload_folder(
    folder_path="tourism-project/deployment",
    repo_id="Neethu-Sathyan/tourism-project",
    repo_type="space"
)
