import json
from collections import Counter
from tqdm import tqdm


def load_jsonl(filepath):
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line)
                data.append(item)
            except json.JSONDecodeError:
                continue
    return data


def basic_quality_check(dataset):
    total = len(dataset)
    bad_format = 0
    short_inputs = 0
    short_outputs = 0
    long_inputs = 0
    long_outputs = 0
    duplicate_inputs = 0
    duplicate_outputs = 0

    input_texts = []
    output_texts = []

    for item in tqdm(dataset, desc="품질 체크 진행중"):
        input_text = item.get("input", "").strip()
        output_text = item.get("output", "").strip()

        # 구조 체크
        if not input_text or not output_text:
            bad_format += 1
            continue

        input_texts.append(input_text)
        output_texts.append(output_text)

        # 길이 체크
        if len(input_text) < 10:
            short_inputs += 1
        if len(output_text) < 10:
            short_outputs += 1
        if len(input_text) > 500:
            long_inputs += 1
        if len(output_text) > 1000:
            long_outputs += 1

    # 중복 체크
    input_counter = Counter(input_texts)
    output_counter = Counter(output_texts)

    duplicate_inputs = sum(1 for v in input_counter.values() if v > 1)
    duplicate_outputs = sum(1 for v in output_counter.values() if v > 1)

    # 결과 출력
    print("\n--- 품질 체크 요약 ---")
    print(f"총 샘플 수: {total}개")
    print(f"잘못된 포맷(빈 input/output) 샘플 수: {bad_format}개")
    print(f"input이 너무 짧은 샘플 수(<10자): {short_inputs}개")
    print(f"output이 너무 짧은 샘플 수(<10자): {short_outputs}개")
    print(f"input이 너무 긴 샘플 수(>500자): {long_inputs}개")
    print(f"output이 너무 긴 샘플 수(>1000자): {long_outputs}개")
    print(f"중복된 input 수: {duplicate_inputs}개")
    print(f"중복된 output 수: {duplicate_outputs}개")
    print("---------------------")


def main():
    filepath = "dataset/raw/dataset.jsonl"  # 확인할 파일 경로
    dataset = load_jsonl(filepath)
    basic_quality_check(dataset)


if __name__ == "__main__":
    main()
