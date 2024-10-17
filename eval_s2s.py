import argparse
import time

import torch

from torch import nn
from torch.utils.data import DataLoader
from transformers import CanineTokenizer

import numpy as np
import pandas as pd

from histocc import (
    OccDatasetV2InMemMultipleFiles,
    load_tokenizer,
    Seq2SeqOccCANINE,
)
from histocc.formatter import (
    BlockyHISCOFormatter,
    BlockyOCC1950Formatter,
    BOS_IDX,
)
from histocc.utils import Averager

from histocc.utils.decoder import greedy_decode

from train_v2 import MAP_FORMATTER


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # File paths, data & model choices
    parser.add_argument('--val-data', type=str, default=None, nargs='+')
    parser.add_argument('--checkpoint', type=str, help='File name of model state')
    parser.add_argument('--fn-out', type=str)
    parser.add_argument('--target-col-naming', type=str, default='hisco')

    # Data parameters
    parser.add_argument('--batch-size', type=int, default=2048)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--pin-memory', action='store_true', default=False)

    # Model parameters
    parser.add_argument('--max-len', type=int, default=128, help='Max. number of characters for input')
    parser.add_argument('--formatter', type=str, default='hisco', choices=MAP_FORMATTER.keys(), help='Target-side tokenization')

    args = parser.parse_args()

    return args


def setup_dataset(
        args: argparse.Namespace,
        formatter: BlockyHISCOFormatter | BlockyOCC1950Formatter,
        tokenizer: CanineTokenizer,
) -> OccDatasetV2InMemMultipleFiles:
    dataset_val = OccDatasetV2InMemMultipleFiles(
        fnames_data=args.val_data,
        formatter=formatter,
        tokenizer=tokenizer,
        max_input_len=args.max_len,
        training=False,
        target_cols=args.target_col_naming,
    )

    return dataset_val


@torch.no_grad
def evaluate(
        model: nn.Module,
        data_loader: torch.utils.data.DataLoader,
        device: torch.device,
        fn_out: str,
):
    model = model.eval()

    inputs = []
    preds_raw = []
    probs_s2s_raw = []
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

        outputs, probs = greedy_decode(
            model=model,
            descr=input_ids,
            input_attention_mask=attention_mask,
            device=device,
            max_len=data_loader.dataset.formatter.max_seq_len,
            start_symbol=BOS_IDX,
            )
        outputs = outputs.detach().cpu().numpy()
        probs = probs.cpu().numpy()

        # Store input in its original string format
        inputs.extend(batch['occ1'])

        # Store predictions and associated "probabilities"
        preds_raw.append(outputs)
        probs_s2s_raw.append(probs)

        # Store labels
        targets = targets.detach().cpu().numpy()
        labels_raw.append(targets)

        batch_time.update(time.time() - end)

        if batch_idx % 10 == 0 or batch_idx == 1:
            print(f'Finished prediction for batch {batch_idx} of {len(data_loader)}')
            print(f'Batch time (data): {batch_time.avg:.2f} ({batch_time_data.avg:.2f}).')
            print(f'Max. memory allocated/reserved: {torch.cuda.max_memory_allocated() / (1024 ** 3):.2f}/{torch.cuda.max_memory_reserved() / (1024 ** 3):.2f} GB')

        end = time.time()

    preds_raw = np.concatenate(preds_raw)
    probs_s2s_raw = np.concatenate(probs_s2s_raw)
    labels_raw = np.concatenate(labels_raw)

    preds = list(map(
        data_loader.dataset.formatter.clean_pred,
        preds_raw,
    ))
    labels = list(map(
        data_loader.dataset.formatter.clean_pred,
        labels_raw,
    ))

    preds = pd.DataFrame({
        'input': inputs,
        'label': labels,
        'pred': preds,
        **{f'prob_{i}': probs_s2s_raw[:, i] for i in range(probs_s2s_raw.shape[1])},
    })

    preds.to_csv(fn_out, index=False)


def main():
    args = parse_args()

    # Target-side tokenization
    formatter = MAP_FORMATTER[args.formatter]()

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

    model = Seq2SeqOccCANINE(
        model_domain='Multilingual_CANINE',
        num_classes=formatter.num_classes,
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
