from transformers import AutoModelForCausalLM
import torch

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-8B",
    device_map="cpu",  # GPU 부족하면 CPU라도 가능
    trust_remote_code=True,
    torch_dtype=torch.bfloat16
)

# 모든 레이어 이름 출력
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):  # Linear 레이어만 필터링
        print(name)
