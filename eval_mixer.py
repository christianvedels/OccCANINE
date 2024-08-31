import argparse
import time

import torch

from torch import nn
from torch.utils.data import DataLoader
from transformers import (
    CanineTokenizer,
)

import numpy as np
import pandas as pd

from histocc import (
    OccDatasetV2InMemMultipleFiles,
    load_tokenizer,
    Seq2SeqMixerOccCANINE,
)
from histocc.formatter import (
    blocky5,
    BlockyHISCOFormatter,
    BOS_IDX,
)
from histocc.utils import Averager

from histocc.utils.decoder import mixer_greedy_decode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # File paths, data & model choices
    parser.add_argument('--val-data', type=str, default=None, nargs='+')
    parser.add_argument('--checkpoint', type=str, help='File name of model state')
    parser.add_argument('--fn-out', type=str)

    # Data parameters
    parser.add_argument('--batch-size', type=int, default=2048)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--pin-memory', action='store_true', default=False)

    # Model parameters
    parser.add_argument('--max-len', type=int, default=128, help='Max. number of characters for input')

    args = parser.parse_args()

    return args


def setup_dataset(
        args: argparse.Namespace,
        formatter: BlockyHISCOFormatter,
        tokenizer: CanineTokenizer,
) -> OccDatasetV2InMemMultipleFiles:
    dataset_val = OccDatasetV2InMemMultipleFiles(
        fnames_data=args.val_data,
        formatter=formatter,
        tokenizer=tokenizer,
        max_input_len=args.max_len,
        training=False,
    )

    return dataset_val


@torch.no_grad
def evaluate(
        model: nn.Module,
        data_loader: torch.utils.data.DataLoader,
        device: torch.device,
        # out_dir: str,
        fn_out: str,
        linear_topk: int = 5,
):
    model = model.eval()

    inputs = []

    preds_s2s_raw = []
    probs_s2s_raw = []

    preds_linear_raw = []
    probs_linear_raw = []

    labels_raw = []

    batch_time = Averager()
    batch_time_data = Averager()

    # Need to initialize first "end time", as this is
    # calculated at bottom of batch loop
    end = time.time()

    for batch_idx, batch in enumerate(data_loader, start=1):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        targets = batch['targets'].to(device)

        batch_time_data.update(time.time() - end)

        outputs_s2s, probs_s2s, outputs_linear, probs_linear = mixer_greedy_decode(
            model=model,
            descr=input_ids,
            input_attention_mask=attention_mask,
            device=device,
            max_len=data_loader.dataset.formatter.max_seq_len,
            start_symbol=BOS_IDX,
            linear_topk=linear_topk,
            )
        outputs_s2s = outputs_s2s.cpu().numpy()
        probs_s2s = probs_s2s.cpu().numpy()
        outputs_linear = outputs_linear.cpu().numpy()
        probs_linear = probs_linear.cpu().numpy()

        # Store input in its original string format
        inputs.extend(batch['occ1'])

        # Store predictions and associated "probabilities"
        preds_s2s_raw.append(outputs_s2s)
        probs_s2s_raw.append(probs_s2s)
        preds_linear_raw.append(outputs_linear)
        probs_linear_raw.append(probs_linear)

        # Store labels
        targets = targets.detach().cpu().numpy()
        labels_raw.append(targets)

        batch_time.update(time.time() - end)

        if batch_idx % 10 == 0 or batch_idx == 1:
            print(f'Finished prediction for batch {batch_idx} of {len(data_loader)}')
            print(f'Batch time (data): {batch_time.avg:.2f} ({batch_time_data.avg:.2f}).')
            print(f'Max. memory allocated/reserved: {torch.cuda.max_memory_allocated() / (1024 ** 3):.2f}/{torch.cuda.max_memory_reserved() / (1024 ** 3):.2f} GB')

        end = time.time()

    preds_s2s_raw = np.concatenate(preds_s2s_raw)
    probs_s2s_raw = np.concatenate(probs_s2s_raw)

    preds_linear_raw = np.concatenate(preds_linear_raw)
    probs_linear_raw = np.concatenate(probs_linear_raw)

    labels_raw = np.concatenate(labels_raw)

    preds_s2s = list(map(
        data_loader.dataset.formatter.clean_pred,
        preds_s2s_raw,
    ))
    preds_linear = np.array([
        [data_loader.dataset.formatter.lookup_hisco[x] for x in y]
        for y in preds_linear_raw
        ])
    labels = list(map(
        data_loader.dataset.formatter.clean_pred,
        labels_raw,
    ))

    preds = pd.DataFrame({
        'input': inputs,
        'label': labels,
        'pred_s2s': preds_s2s,
        **{f'pred_linear_{i}': preds_linear[:, i] for i in range(linear_topk)},
        **{f'prob_s2s_{i}': probs_s2s_raw[:, i] for i in range(probs_s2s_raw.shape[1])},
        **{f'prob_linear_{i}': probs_linear_raw[:, i] for i in range(linear_topk)},
    })

    preds.to_csv(fn_out, index=False)


def main():
    args = parse_args()

    # Target-side tokenization
    formatter = blocky5()

    # Input-side tokenization
    tokenizer = load_tokenizer(
        model_domain='Multilingual_CANINE',
    )

    # Dataset
    dataset_val = setup_dataset(
        args=args,
        formatter=formatter,
        tokenizer=tokenizer,
    )

    # Data loader
    data_loader_val = DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        )

    # Setup model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = Seq2SeqMixerOccCANINE(
        model_domain='Multilingual_CANINE', # TODO make arg, discuss with Vedel
        num_classes=formatter.num_classes,
        num_classes_flat=1919, # TODO make arg or infer from formatter
    ).to(device)

    model.load_state_dict(torch.load(args.checkpoint)['model'])

    evaluate(
        model=model,
        data_loader=data_loader_val,
        device=device,
        fn_out=args.fn_out,
        )


if __name__ == '__main__':
    main()
