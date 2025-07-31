import os
import datasets
import huggingface_hub

local_dir = "data"  

# Everything except the PDFs
repliqa = datasets.load_dataset("ServiceNow/repliqa")

# get all PDFs at once
snapshot_path = huggingface_hub.snapshot_download(repo_id="ServiceNow/repliqa", repo_type="dataset", local_dir=local_dir)

def get_path_to_local_pdf_snapshot(sample: dict[str, str]) -> str:
    return os.path.join(snapshot_path, sample["document_path"])

path_to_local_pdf_snapshot = get_path_to_local_pdf_snapshot(repliqa["repliqa_0"][0])
path_to_local_pdf_snapshot