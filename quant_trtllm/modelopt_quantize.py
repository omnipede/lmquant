import modelopt.torch.quantization as mtq
from modelopt.torch.export import export_hf_checkpoint

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = 'google/gemma-4-31B-it'
OUTPUT_DIR = './models'

def main():
    # Load the model from HuggingFace
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    # Select the quantization config, for example, FP8
    config = mtq.FP8_PER_CHANNEL_PER_TOKEN_CFG

    # PTQ with in-place replacement of quantized modules
    model = mtq.quantize(model, config)

    SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-TRT_FP8"
    SAVE_DIR = f"{OUTPUT_DIR}/{SAVE_DIR}"
    with torch.inference_mode():
        export_hf_checkpoint(model, export_dir=SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)


if __name__ == '__main__':
    main()
