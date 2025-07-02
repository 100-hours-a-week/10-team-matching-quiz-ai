from datasets import load_dataset
import json

def load_dataset_from_jsonl(path: str):
    raw_dataset = load_dataset("json", data_files={"train": path})["train"]

    def format_example(example):
        # input에 이미 "[질문 목록] ..." 형태로 되어 있음
        prompt = example["input"]
        # output도 바로 사용 (json string 형태로 변환)
        completion = json.dumps(example["output"], ensure_ascii=False)
        return {
            "prompt": prompt,
            "completion": completion
        }

    return raw_dataset.map(format_example)
