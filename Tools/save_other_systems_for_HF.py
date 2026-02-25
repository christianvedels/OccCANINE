"""
Save OCCICEM, ISCO68, and OCC1950 models in HuggingFace-compatible format.

Config is inferred directly from state dict tensor shapes, so no manual
specification of num_classes or vocab sizes is required.

All three models use Seq2SeqMixerOccCANINE_hub with:
    - block_size = 3, max_num_codes = 2  →  seq_len = 8
(This matches the formatter used during training for each system.)

Example use:
    python Tools/save_other_systems_for_HF.py
    python Tools/save_other_systems_for_HF.py --test
    python Tools/save_other_systems_for_HF.py --push-to-hub
"""

import argparse
import os
import sys

# Ensure the repo root is on the path so histocc resolves as a local package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from histocc.model_assets import Seq2SeqMixerOccCANINE_hub


# Seq len = block_size * max_num_codes + 2 (BOS + EOS)
# All three other-system models: block_size=3, max_num_codes=2 → seq_len=8
_SEQ_LEN = 8

MODELS = [
    {
        "hf_name": "OccCANINE_OCCICEM",
        "fn_in": "Data/models/mixer-icem-ft/last.bin",
        "fn_out": "Data/models/OccCANINE_OCCICEM_forHF",
        "seq_len": _SEQ_LEN,
    },
    {
        "hf_name": "OccCANINE_ISCO68",
        "fn_in": "Data/models/mixer-isco-ft/last.bin",
        "fn_out": "Data/models/OccCANINE_ISCO68_forHF",
        "seq_len": _SEQ_LEN,
    },
    {
        "hf_name": "OccCANINE_OCC1950",
        "fn_in": "Data/models/mixer-occ1950-ft/last.bin",
        "fn_out": "Data/models/OccCANINE_OCC1950_forHF",
        "seq_len": _SEQ_LEN,
    },
]


def infer_config(state_dict: dict, seq_len: int) -> dict:
    """Infer model config from state dict tensor shapes."""
    vocab_size = state_dict["decoder.head.weight"].shape[0]
    num_classes_flat = state_dict["linear_decoder.weight"].shape[0]

    return {
        "model_domain": "Multilingual_CANINE",
        "model_type": "seq2seq_mixer",
        "num_classes": [vocab_size - 1] * seq_len,
        "num_classes_flat": num_classes_flat,
        "dropout_rate": 0,
    }


def save_model(fn_in: str, fn_out: str, seq_len: int) -> dict:
    loaded = torch.load(fn_in, weights_only=False)
    state_dict = loaded["model"]

    config = infer_config(state_dict, seq_len)

    model = Seq2SeqMixerOccCANINE_hub(config)
    model.load_state_dict(state_dict)
    model.save_pretrained(fn_out, config=config)

    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test",
        action="store_true",
        default=False,
        help="After saving, verify each model can be loaded back from HF.",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        default=False,
        help="Push saved models to HuggingFace Hub under Christianvedel/<hf_name>.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    for m in MODELS:
        print(f"\n--- {m['hf_name']} ---")
        print(f"  Loading from : {m['fn_in']}")
        print(f"  Saving to    : {m['fn_out']}")

        config = save_model(m["fn_in"], m["fn_out"], m["seq_len"])

        print(f"  vocab_size      : {max(config['num_classes']) + 1}")
        print(f"  num_classes_flat: {config['num_classes_flat']}")
        print(f"  seq_len         : {len(config['num_classes'])}")
        print(f"  Saved OK.")

        if args.push_to_hub:
            hf_repo = f"Christianvedel/{m['hf_name']}"
            print(f"  Pushing to HF: {hf_repo}")
            model = Seq2SeqMixerOccCANINE_hub.from_pretrained(m["fn_out"])
            model.push_to_hub(hf_repo)
            print(f"  Push OK.")

        if args.test:
            hf_repo = f"Christianvedel/{m['hf_name']}"
            print(f"  Testing round-trip from HF: {hf_repo}")
            Seq2SeqMixerOccCANINE_hub.from_pretrained(hf_repo, force_download=True)
            print(f"  Round-trip OK.")


if __name__ == "__main__":
    main()
