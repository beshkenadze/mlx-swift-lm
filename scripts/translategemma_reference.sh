#!/usr/bin/env bash
#
# TranslateGemma reference-parity check (CUDA / transformers).
#
# Produces a reference translation from the canonical `transformers` implementation, using
# the SAME structured chat template (source/target language codes) and greedy decode as the
# Swift integration test (`ChatSessionTests.translation`, mlx-swift-lm). Compare the output
# below against the Swift output to confirm the Swift Gemma 3 port translates correctly.
#
# Why transformers+CUDA instead of mlx-lm: the MLX core wheel does not load on this Linux/WSL
# box (`libmlx.so` is not resolvable), so we use the original full-precision weights on the
# GPU. NOTE: the Swift side runs the 4-bit MLX quant; under greedy decode a short sentence
# should match, but minor wording differences from quantization are acceptable.
#
# Run on pc.lan (WSL, RTX 4090) inside tmux — never run heavy jobs over a bare SSH session:
#
#   ssh pc.lan
#   tmux new -s tg                 # or: tmux attach -t tg
#   bash ~/translategemma_reference.sh
#
# Requirements (already present on pc.lan): uv, a working CUDA torch in the system Python,
# and a Hugging Face token in ~/.cache/huggingface (google/translategemma-* is gated).

set -euo pipefail

MODEL="google/translategemma-4b-it"
SRC_LANG="en"
TGT_LANG="fr"
TEXT="Hello, how are you?"      # MUST match the Swift integration test input
MAX_TOKENS=200
VENV="$HOME/.tg_ref_venv"

export PATH="$HOME/.local/bin:$PATH"

echo "=================================================="
echo "Model : ${MODEL} (full precision, CUDA)"
echo "Lang  : ${SRC_LANG} -> ${TGT_LANG}"
echo "Text  : ${TEXT}"
echo "Decode: greedy (do_sample=False), max_new_tokens ${MAX_TOKENS}"
echo "=================================================="

# Reuse the system CUDA torch (--system-site-packages); add transformers via uv (no pip).
uv venv --system-site-packages "${VENV}" >/dev/null 2>&1 || true
uv pip install --python "${VENV}/bin/python" "transformers>=4.57.0" accelerate pillow

MODEL="${MODEL}" SRC_LANG="${SRC_LANG}" TGT_LANG="${TGT_LANG}" TEXT="${TEXT}" \
  MAX_TOKENS="${MAX_TOKENS}" "${VENV}/bin/python" - <<'PY'
import os, torch
from transformers import AutoProcessor, AutoModelForImageTextToText

model_id = os.environ["MODEL"]
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForImageTextToText.from_pretrained(
    model_id, device_map="auto", torch_dtype=torch.bfloat16
)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "source_lang_code": os.environ["SRC_LANG"],
                "target_lang_code": os.environ["TGT_LANG"],
                "text": os.environ["TEXT"],
            }
        ],
    }
]

inputs = processor.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True,
    return_dict=True, return_tensors="pt",
).to(model.device, dtype=torch.bfloat16)
input_len = inputs["input_ids"].shape[-1]

with torch.inference_mode():
    generation = model.generate(**inputs, max_new_tokens=int(os.environ["MAX_TOKENS"]), do_sample=False)

decoded = processor.decode(generation[0][input_len:], skip_special_tokens=True)
print("\n=================== REFERENCE OUTPUT ===================")
print(decoded.strip())
print("=======================================================")
PY

echo
echo "Compare the REFERENCE OUTPUT above with the Swift integration-test output"
echo "(TranslateGemmaIntegrationTests). Greedy decode should give the same/equivalent French."
