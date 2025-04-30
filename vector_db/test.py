import pandas as pd
import re

# CSV 파일 경로
csv_path = "/Users/david.lee/Library/CloudStorage/GoogleDrive-yusc5841@gmail.com/내 드라이브/KTB/github_ai/10-team-matching-quiz-ai/vector_db/questions.csv"

# CSV 파일 읽기
df = pd.read_csv(csv_path)

# 'questions' 컬럼이 존재하는지 확인
if "questions" not in df.columns:
    raise ValueError("'questions' 컬럼이 없습니다.")

# 마침표(.)와 물음표(?) 기준으로 분리
all_questions = []
for q in df["questions"]:
    if isinstance(q, str):
        split_qs = re.split(r'[.?]', q)  # 정규표현식으로 . 또는 ? 기준 분리
        cleaned_qs = [s.strip() for s in split_qs if s.strip()]  # 공백 제거 + 빈 문자열 제거
        all_questions.extend(cleaned_qs)

# 데이터프레임으로 변환
question_df = pd.DataFrame({'question': all_questions})

# 결과 확인
print(f"총 질문 수: {len(question_df)}")
print(question_df.head())

# CSV로 저장
question_df.to_csv("questions_split_by_dot_or_qmark.csv", index=False)
