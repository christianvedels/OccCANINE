# Occupational CANINE: HISCO Classification Model (Seq2Seq + Mixer)

## Overview

OccCANINE_s2s_mix is the recommended version of [OccCANINE](https://huggingface.co/Christianvedel/OccCANINE). It combines a [CANINE](https://huggingface.co/google/canine-s) encoder with a sequential decoder trained using a mixed loss that blends sequence-level and flat classification objectives. This is the default model loaded by the `histocc` package and achieves around 96% F1 at the 5-digit HISCO level.

See more on: [GitHub.com/christianvedels/OccCANINE](https://github.com/christianvedels/OccCANINE)

Read the paper on arXiv: [https://arxiv.org/abs/2402.13604](https://arxiv.org/abs/2402.13604)

## Key Features

- **High Accuracy**: Around 96% F1 at the 5-digit HISCO level.
- **Multilingual Support**: Trained on 15.8 million description-HISCO code pairs across 13 languages.
- **Sequential decoding**: Outputs full HISCO codes digit-by-digit, naturally respecting the hierarchical structure of the classification system.
- **Mixed loss training**: Combines sequence-level and flat classification losses, improving both precision and recall.

## Usage

```python
from histocc import OccCANINE

# OccCANINE_s2s_mix is loaded by default
model = OccCANINE()

result = model.predict("blacksmith", lang="en")
```

The model is also accessible via the command-line interface:

```bash
python predict.py --fn-in path/to/input.csv --col occ1 --fn-out path/to/output.csv --language en
```

See [GETTING_STARTED.md](https://github.com/christianvedels/OccCANINE/blob/main/GETTING_STARTED.md) for a full installation and usage guide.

## Supported Languages

English (`en`), Danish (`da`), Swedish (`se`), Dutch (`nl`), Catalan (`ca`), French (`fr`), Norwegian (`no`), Icelandic (`is`), Portuguese (`pt`), German (`ge`), Spanish (`es`), Italian (`it`), Greek (`gr`).

## Contribution and Support

Developed at the University of Southern Denmark by Christian Møller Dahl, Torben Johansen and Christian Vedel.

---

**Model Details:**
- **Task**: Text Classification / Sequence Generation
- **Base Model**: CANINE
- **Framework**: Transformers / PyTorch
- **Languages**: 13 languages
- **License**: Apache 2.0
- **Paper**: arXiv 2402.13604
