import json


def build_text_field(input_text, output_text):
    return f"""<|im_start|>user
당신은 카부캠 교육생들을 대상으로 모의 면접을 진행하는 AI 면접관입니다.

아래 정보를 참고하여,
- 해당 대표 질문에 이어질 수 있는 꼬리 질문 4개를 생성하세요.

# 조건
- 꼬리 질문들은 서로 다른 관점(개념/구현/문제 해결/최신 트렌드 등)을 다뤄야 합니다.
- 결과만 출력하세요.

# 입력 정보
{input_text}

# 출력 포맷
꼬리 질문:
{output_text}
<|im_end|>
"""


input_data = "dataset/raw/dataset.jsonl"
output_data = "dataset/raw/qwen_dataset.jsonl"

with open(input_data, 'r', encoding='utf-8') as infile, \
        open(output_data, 'w', encoding='utf-8') as outfile:
    for line in infile:
        data = json.loads(line)
        input_text = data.get('input', '').strip()
        output_text = data.get('output', '').strip()
        formatted = {'text': build_text_field(input_text, output_text)}
        outfile.write(json.dumps(formatted, ensure_ascii=False)+'\n')

print('finished')
