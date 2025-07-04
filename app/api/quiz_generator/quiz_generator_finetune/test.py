import json

input_path = "quiz_finetune_final.jsonl"
output_path = "quiz_finetune_final_dedup.jsonl"

seen_inputs = set()
count_total = 0
count_written = 0

with open(input_path, "r", encoding="utf-8") as infile, \
     open(output_path, "w", encoding="utf-8") as outfile:
    for line in infile:
        try:
            data = json.loads(line)
        except Exception as e:
            print("⚠️ JSON decode error:", e)
            continue
        # input 기준으로만 중복 제거
        key = data.get("input")
        if key is None:
            continue  # input이 없는 라인은 스킵
        count_total += 1
        if key not in seen_inputs:
            seen_inputs.add(key)
            outfile.write(json.dumps(data, ensure_ascii=False) + "\n")
            count_written += 1

print(f"✅ 중복 제거 완료: {count_total}줄 중 {count_written}줄만 남김!")
