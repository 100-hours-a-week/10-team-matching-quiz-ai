# Qwen3-8B LoRA Fine-Tuning

이 프로젝트는 Qwen3-8B 모델을 LoRA 방식과 4bit 양자화로 미세 조정(SFT)하는 코드입니다.

## 폴더 구조

```
├── config.py
├── dataset.py
├── train.py
├── requirements.txt
├── quiz_finetune.jsonl
```

## 실행 방법

```bash
pip install -r requirements.txt
python train.py
```

## 결과 저장
- 학습된 모델은 `./qwen3_lora_output_<timestamp>`에 저장됩니다.
- TensorBoard로 학습 로그 추적 가능 (`tensorboard --logdir=./qwen3_lora_output_*/runs`)