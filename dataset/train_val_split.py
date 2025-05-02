import json
import random
from sklearn.model_selection import train_test_split
import os


def split_jsonl(input_file, train_file, val_file, train_ratio=0.8, seed=42):
    """
    JSONL 파일을 train과 validation 세트로 나눕니다.

    Args:
        input_file (str): 입력 JSONL 파일 경로
        train_file (str): 출력할 train JSONL 파일 경로
        val_file (str): 출력할 validation JSONL 파일 경로
        train_ratio (float): 학습 데이터 비율 (기본값: 0.8)
        seed (int): 랜덤 시드 값
    """
    # 데이터 로드
    data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            # 빈 줄이나 주석 건너뛰기
            line = line.strip()
            if not line or line.startswith('//'):
                continue

            try:
                # JSON 파싱 시도
                item = json.loads(line)
                data.append(item)
            except json.JSONDecodeError as e:
                print(f"줄 {line_num}에서 JSON 파싱 오류 발생: {e}")
                print(f"문제가 된 줄: {line[:100]}...")  # 첫 100자만 출력
                # 계속 진행하려면 pass, 중단하려면 raise를 사용

    if not data:
        raise ValueError("파싱된 데이터가 없습니다. 파일 형식을 확인하세요.")

    # 출력 디렉토리 확인 및 생성
    os.makedirs(os.path.dirname(train_file), exist_ok=True)
    os.makedirs(os.path.dirname(val_file), exist_ok=True)

    # 데이터 분할
    train_data, val_data = train_test_split(
        data,
        train_size=train_ratio,
        random_state=seed,
        shuffle=True
    )

    # 학습 데이터 저장
    with open(train_file, 'w', encoding='utf-8') as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    # 검증 데이터 저장
    with open(val_file, 'w', encoding='utf-8') as f:
        for item in val_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"전체 데이터 수: {len(data)}")
    print(f"학습 데이터 수: {len(train_data)} ({train_ratio*100:.1f}%)")
    print(f"검증 데이터 수: {len(val_data)} ({(1-train_ratio)*100:.1f}%)")


if __name__ == "__main__":
    # 파일 경로 설정
    input_file = "dataset/raw/qwen_dataset.jsonl"  # 입력 JSONL 파일
    train_file = "dataset/processed/train.jsonl"  # 학습 데이터 출력 파일
    val_file = "dataset/processed/val.jsonl"  # 검증 데이터 출력 파일

    # 데이터 분할 실행
    split_jsonl(input_file, train_file, val_file, train_ratio=0.8, seed=42)
