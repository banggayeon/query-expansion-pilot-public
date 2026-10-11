"""Download only the approved MiniLM CPU export; never performs inference."""
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
os.environ["HF_HOME"] = str(ROOT / ".model-cache")
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"


def main():
    from huggingface_hub import HfApi, hf_hub_download
    model = "sentence-transformers/all-MiniLM-L6-v2"
    start = time.perf_counter()
    revision = HfApi(token=False).model_info(model).sha
    files = {}
    for filename in ("onnx/model.onnx", "tokenizer.json", "config.json", "sentence_bert_config.json", "1_Pooling/config.json"):
        files[filename] = hf_hub_download(model, filename, revision=revision,
                                         cache_dir=str(ROOT / ".model-cache/hub"), token=False)
    print(json.dumps({"model_id": model, "revision": revision, "files": files,
                      "download_wall_seconds": time.perf_counter() - start}, indent=2))


if __name__ == "__main__":
    main()
