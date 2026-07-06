from transformers import AutoModelForCausalLM, AutoProcessor, AutoModelForMultimodalLM
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

# Reference
# https://github.com/vllm-project/llm-compressor/blob/main/examples/quantization_w8a8_fp8/README.md

MODEL_ID = 'google/gemma-4-31B-it'
OUTPUT_DIR = './models'


def main():
    model = AutoModelForMultimodalLM.from_pretrained(MODEL_ID)
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    # Configure the simple PTQ quantization
    recipe = QuantizationModifier(
        targets="Linear", 
        scheme="FP8_DYNAMIC", 
        ignore=[
            "re:.*vision.*",
            "re:.*embed_tokens.*",
            "lm_head",
        ],
    )

    # Apply the quantization algorithm.
    oneshot(model=model, recipe=recipe)

    # Save the model.
    SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-FP8-Dynamic"
    SAVE_DIR = f"{OUTPUT_DIR}/{SAVE_DIR}"
    model.save_pretrained(SAVE_DIR)
    processor.save_pretrained(SAVE_DIR)


if __name__ == '__main__':
    main()
