from datasets import load_dataset
import json

def load_dataset_from_jsonl(path: str):
    raw_dataset = load_dataset("json", data_files={"train": path})["train"]

    def format_example(example):
        # [질문 목록] + 개행 + 질문들을 한 줄씩 넣는 구조로 prompt 생성
        joined_questions = '\n'.join(example["question_list"])
        prompt = "[질문 목록]\n" + joined_questions

        # output은 json string으로 저장 (기존과 동일)
        completion = json.dumps(example["output"], ensure_ascii=False)
        return {
            "prompt": prompt,
            "completion": completion
        }

    return raw_dataset.map(format_example)
