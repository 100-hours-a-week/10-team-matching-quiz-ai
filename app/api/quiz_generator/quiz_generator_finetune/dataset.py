from datasets import load_dataset

def load_dataset_from_jsonl(path: str):
    raw_dataset = load_dataset("json", data_files={"train": path})["train"]

    def format_example(example):
        prompt = example["instruction"]
        if example.get("input"):
            prompt += f"\n\n{example['input']}"
        return {
            "prompt": prompt,
            "completion": example["output"]
        }

    formatted_dataset = raw_dataset.map(format_example)
    return formatted_dataset