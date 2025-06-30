from datasets import load_dataset
import json

def load_dataset_from_jsonl(path: str):
    raw_dataset = load_dataset("json", data_files={"train": path})["train"]

    def format_example(example):
        return {
            "prompt": example["input"],
            "completion": json.dumps(example["output"], ensure_ascii=False)
        }

    return raw_dataset.map(format_example)
