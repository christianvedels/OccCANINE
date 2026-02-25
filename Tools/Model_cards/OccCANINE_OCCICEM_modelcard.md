# OccCANINE: I-CeM Occupational Classification (OCCICEM)

## Overview

OccCANINE_OCCICEM is a version of [OccCANINE](https://github.com/christianvedels/OccCANINE) fine-tuned to automatically convert English occupational descriptions into [I-CeM](https://www.essex.ac.uk/research-projects/integrated-census-microdata) (Integrated Census Microdata) occupational codes. It uses a CANINE encoder with a sequential decoder trained using a mixed loss, fine-tuned from the [OccCANINE_s2s_mix](https://huggingface.co/Christianvedel/OccCANINE_s2s_mix) base model on IPUMS UK census data.

See more on: [GitHub.com/christianvedels/OccCANINE](https://github.com/christianvedels/OccCANINE)

Read the paper on arXiv: [https://arxiv.org/abs/2402.13604](https://arxiv.org/abs/2402.13604)

## Key Features

- **English**: Trained and evaluated on English occupational descriptions.
- **Sequential decoding**: Outputs I-CeM codes digit-by-digit.
- **Mixed loss training**: Combines sequence-level and flat classification losses.
- **Fine-tuned**: Initialized from OccCANINE_s2s_mix and fine-tuned on IPUMS UK I-CeM data.

## Usage

```python
from histocc import OccCANINE

model = OccCANINE(name="OccCANINE_OCCICEM", system="OCCICEM", hf=True)

result = model.predict("blacksmith", lang="en")
```

## Contribution and Support

Developed at the University of Southern Denmark by Christian Møller Dahl, Torben Johansen and Christian Vedel.

---

**Model Details:**
- **Task**: Text Classification / Sequence Generation
- **Base Model**: CANINE (fine-tuned from OccCANINE_s2s_mix)
- **Target system**: I-CeM (OCCICEM)
- **Language**: English
- **Framework**: Transformers / PyTorch
- **License**: Apache 2.0
- **Paper**: arXiv 2402.13604
