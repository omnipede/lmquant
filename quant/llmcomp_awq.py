import torch
from transformers import AutoProcessor, AutoModelForMultimodalLM
from llmcompressor import oneshot
from llmcompressor.modifiers.transform import AWQModifier
from llmcompressor.modifiers.quantization import QuantizationModifier
from datasets import load_dataset


# Reference
# https://huggingface.co/ebircak/gemma-4-31B-it-4bit-W4A16-AWQ/blob/main/recipe.yaml

MODEL_ID = 'google/gemma-4-31B-it'
OUTPUT_DIR = './models'

CALIB_DS_ID = 'HuggingFaceH4/ultrachat_200k'
CALIB_DS_SPLIT = 'train_sft'
CALIB_MAX_SEQ_LENGTH = 4096
CALIB_NUM_SAMPLES = 512


def main():
    model = AutoModelForMultimodalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, 
        device_map='auto',
        max_memory={
            0: "70GiB",
            1: "70GiB",
            2: "70GiB",
            3: "70GiB",
            4: "70GiB",
            5: "70GiB",
            6: "70GiB",
            7: "70GiB",
        },
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    tokenizer = processor.tokenizer

    def preproc(example):
        return {
            "text": tokenizer.apply_chat_template(
                example["messages"],
                tokenize=False,
            )
        }
    calib_dataset = load_dataset(CALIB_DS_ID, split=CALIB_DS_SPLIT).shuffle(seed=42).select(range(CALIB_NUM_SAMPLES))
    calib_dataset = calib_dataset.map(preproc, num_proc=16)

    recipe = [
        AWQModifier(
            mappings=build_gemma4_awq_mappings(num_layers=60),
        ),
        QuantizationModifier(
            targets=["Linear"],
            scheme="W4A16_ASYM",
            ignore=[
                "re:.*vision.*",
                "re:.*embed_tokens.*",
                "re:.*multi_modal_projector.*",
                "lm_head",
            ],
        ),
    ]

    oneshot(
        model=model,
        tokenizer=processor.tokenizer,
        dataset=calib_dataset,
        recipe=recipe,
        max_seq_length=CALIB_MAX_SEQ_LENGTH,    
        num_calibration_samples=CALIB_NUM_SAMPLES,
        sequential_targets=['Linear'],
    )

    # Save the model.
    SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-AWQ"
    SAVE_DIR = f"{OUTPUT_DIR}/{SAVE_DIR}"
    model.save_pretrained(SAVE_DIR)
    processor.save_pretrained(SAVE_DIR)


def build_gemma4_awq_mappings(num_layers=60):
    mappings = []

    for i in range(num_layers):
        layer = rf"re:.*language_model\.layers\.{i}"

        if i in [5, 11, 17, 23, 29, 35, 41, 47, 53, 59]:
            mappings.append({
                "smooth_layer": rf"{layer}\.input_layernorm$",
                "balance_layers": [
                    rf"{layer}\.self_attn\.q_proj$",
                    rf"{layer}\.self_attn\.k_proj$",
                ],
                "activation_hook_target": None,
            })
        else:
            mappings.append({
                "smooth_layer": rf"{layer}\.input_layernorm$",
                "balance_layers": [
                    rf"{layer}\.self_attn\.q_proj$",
                    rf"{layer}\.self_attn\.k_proj$",
                    rf"{layer}\.self_attn\.v_proj$",
                ],
                "activation_hook_target": None,
            })

        mappings.append({
            "smooth_layer": rf"{layer}\.pre_feedforward_layernorm$",
            "balance_layers": [
                rf"{layer}\.mlp\.gate_proj$",
                rf"{layer}\.mlp\.up_proj$",
            ],
            "activation_hook_target": None,
        })

        mappings.append({
            "smooth_layer": rf"{layer}\.mlp\.up_proj$",
            "balance_layers": [
                rf"{layer}\.mlp\.down_proj$",
            ],
            "activation_hook_target": None,
        })

    return mappings


if __name__ == '__main__':
    main()
