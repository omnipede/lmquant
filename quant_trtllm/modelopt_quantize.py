import json
import os
import shutil
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

import modelopt.torch.quantization as mtq
from modelopt.torch.export import export_tensorrt_llm_checkpoint


MODEL_ID = 'google/gemma-4-31B-it'
OUTPUT_DIR = './models'

def main():
    TEXT_ONLY_DIR = Path(f"{OUTPUT_DIR}/{MODEL_ID.rstrip("/").split("/")[-1] + "-TEXT_ONLY"}")
    # convert_to_text_only(MODEL_ID, TEXT_ONLY_DIR)
    # Load the model from HuggingFace
    model = AutoModelForCausalLM.from_pretrained(TEXT_ONLY_DIR, dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(TEXT_ONLY_DIR)

    # Select the quantization config, for example, FP8
    config = mtq.FP8_PER_CHANNEL_PER_TOKEN_CFG

    # PTQ with in-place replacement of quantized modules
    model = mtq.quantize(model, config)

    SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-TRT_FP8"
    SAVE_DIR = f"{OUTPUT_DIR}/{SAVE_DIR}"
    with torch.inference_mode():
        export_tensorrt_llm_checkpoint(
            model, 
            "gemma", 
            export_dir=SAVE_DIR,
            inference_tensor_parallel=2,
            inference_pipeline_parallel=1,
        )
    tokenizer.save_pretrained(SAVE_DIR)


def convert_to_text_only(src_model_id: str, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    # 전체 checkpoint 다운로드
    src_dir = Path(snapshot_download(src_model_id))

    # 루트 Gemma4Config 안의 text_config만 추출
    full_config = AutoConfig.from_pretrained(src_dir)
    text_config = full_config.text_config

    # AutoModelForCausalLM이 Gemma4ForCausalLM로 잡히도록 명시
    text_config.architectures = ["Gemma4ForCausalLM"]
    text_config.save_pretrained(out_dir)

    # tokenizer/processor 관련 파일 복사
    copy_tokenizer_files(src_dir, out_dir)

    # 원본 safetensors index 확인
    index_path = src_dir / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path, "r") as f:
            index = json.load(f)
        shard_files = sorted(set(index["weight_map"].values()))
    else:
        shard_files = [p.name for p in sorted(src_dir.glob("*.safetensors"))]

    new_weight_map = {}
    total_size = 0
    new_shard_idx = 0

    for shard_name in shard_files:
        shard_path = src_dir / shard_name
        tensors = {}

        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                new_key = None

                # multimodal wrapper 내부의 text model:
                # model.language_model.layers... -> model.layers...
                if key.startswith("model.language_model."):
                    new_key = "model." + key[len("model.language_model."):]

                # lm_head는 보통 top-level에 있음
                elif key.startswith("lm_head."):
                    new_key = key

                if new_key is not None:
                    tensor = f.get_tensor(key)
                    tensors[new_key] = tensor
                    total_size += tensor.numel() * tensor.element_size()

        if not tensors:
            continue

        new_shard_idx += 1
        out_shard_name = f"model-{new_shard_idx:05d}-of-XXXXX.safetensors"
        out_shard_path = out_dir / out_shard_name
        save_file(tensors, out_shard_path)

        for k in tensors:
            new_weight_map[k] = out_shard_name

    # shard 파일명에서 전체 shard 개수 반영
    final_weight_map = {}
    shard_count = new_shard_idx

    for old_name in sorted(set(new_weight_map.values())):
        new_name = old_name.replace("XXXXX", f"{shard_count:05d}")
        os.rename(out_dir / old_name, out_dir / new_name)

    for k, old_name in new_weight_map.items():
        final_weight_map[k] = old_name.replace("XXXXX", f"{shard_count:05d}")

    with open(out_dir / "model.safetensors.index.json", "w") as f:
        json.dump(
            {
                "metadata": {"total_size": total_size},
                "weight_map": final_weight_map,
            },
            f,
            indent=2,
        )

    print(f"Saved text-only checkpoint to: {out_dir}")


def copy_tokenizer_files(src_dir: Path, out_dir: Path):
    patterns = [
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "chat_template.json",
        "chat_template.jinja",
        "generation_config.json",
    ]

    for name in patterns:
        src = src_dir / name
        if src.exists():
            shutil.copy2(src, out_dir / name)


if __name__ == '__main__':
    main()
