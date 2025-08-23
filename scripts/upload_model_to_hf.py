from huggingface_hub import HfApi
import os
from dotenv import load_dotenv

load_dotenv()

api = HfApi(token=os.getenv("HF_TOKEN"))
api.upload_folder(
    folder_path="./models/best_sms_content_model",
    repo_id="putthaimastan/wangchanberta_thai_sms_content_classification",
    repo_type="model",
)