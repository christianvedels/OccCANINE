import argparse
import os
import yaml

import torch

from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    get_linear_schedule_with_warmup,
    CanineTokenizer,
)

from histocc import (
    OccDatasetMixerInMemMultipleFiles,
    load_tokenizer,
    Seq2SeqMixerOccCANINE,
    BlockOrderInvariantLoss,
    LossMixer,
    CANINEOccupationClassifier_hub,
)
from histocc.seq2seq_mixer_engine import train
from histocc.formatter import (
    blocky5,
    BlockyHISCOFormatter,
    PAD_IDX,
)
from histocc.utils import wandb_init

try:
    # want to do import to set has_wandb even if not used directly
    import wandb # pylint: disable=W0611
    has_wandb = True # pylint: disable=C0103
except ImportError:
    has_wandb = False # pylint: disable=C0103

# TODO torch.cudnn.benchmark

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # File paths, data & model choices
    parser.add_argument('--save-dir', type=str, default=None)
    parser.add_argument('--save-interval', type=int, default=5000, help='Number of steps between saving model')
    parser.add_argument('--initial-checkpoint', type=str, default=None, help='Model weights to use for initialization. Discarded if resume state exists at --save-dir')

    parser.add_argument('--train-data', type=str, default=None, nargs='+')
    parser.add_argument('--val-data', type=str, default=None, nargs='+')

    # Logging parameters
    parser.add_argument('--log-interval', type=int, default=100, help='Number of steps between reporting training stats')
    parser.add_argument('--eval-interval', type=int, default=1000, help='Number of steps between calculating and logging validation performance')
    parser.add_argument('--log-wandb', action='store_true', default=False, help='Whether to log validation performance using W&B')

    # Data parameters
    parser.add_argument('--num-epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--pin-memory', action='store_true', default=False)

    # Model and optimizer parameters
    parser.add_argument('--learning-rate', type=float, default=2e-05)
    parser.add_argument('--warmup-steps', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0.0, help='Classifier dropout rate. Does not affect encoder.')
    parser.add_argument('--max-len', type=int, default=128, help='Max. number of characters for input')
    parser.add_argument('--decoder-dim-feedforward', type=int, default=None, help='Defaults to endoder hidden dim if not specified.')
    parser.add_argument('--seq2seq-weight', type=float, default=0.5)

    # Augmentation
    parser.add_argument('--num-transformations', type=int, default=3)
    parser.add_argument('--augmentation-prob', type=float, default=0.3)
    parser.add_argument('--unk-lang-prob', type=float, default=0.25)

    args = parser.parse_args()

    if args.log_wandb and not has_wandb:
        raise ImportError('Specified --log-wandb, but wandb is not installed')

    # TODO add checks on file paths and directories

    return args


def setup_datasets(
        args: argparse.Namespace,
        formatter: BlockyHISCOFormatter,
        tokenizer: CanineTokenizer,
) -> tuple[OccDatasetMixerInMemMultipleFiles, OccDatasetMixerInMemMultipleFiles]:
    dataset_train = OccDatasetMixerInMemMultipleFiles(
        fnames_data=args.train_data,
        formatter=formatter,
        tokenizer=tokenizer,
        max_input_len=args.max_len,
        num_classes_flat=1919,
        training=True,
        alt_prob=args.augmentation_prob,
        n_trans=args.num_transformations,
        unk_lang_prob=args.unk_lang_prob,
    )

    dataset_val = OccDatasetMixerInMemMultipleFiles(
        fnames_data=args.val_data,
        formatter=formatter,
        tokenizer=tokenizer,
        max_input_len=args.max_len,
        num_classes_flat=1919,
        training=False,
    )

    return dataset_train, dataset_val


def load_states(
        save_dir: str,
        model: Seq2SeqMixerOccCANINE,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        initial_checkpoint: str | None = None,
) -> int:
    if 'last.bin' in os.listdir(save_dir):
        print(f'Model states exist at {save_dir}. Resuming from last.bin')

        states = torch.load(os.path.join(save_dir, 'last.bin'))

        model.load_state_dict(states['model'])
        optimizer.load_state_dict(states['optimizer'])
        scheduler.load_state_dict(states['scheduler'])

        current_step = states['step']

        return current_step

    if initial_checkpoint is None:
        return 0

    if initial_checkpoint.lower() == 'occ-canine-v1':
        print('Initializing encoder from HF (christianvedel/OccCANINE)')
        encoder = CANINEOccupationClassifier_hub.from_pretrained("christianvedel/OccCANINE")
        model.encoder.load_state_dict(encoder.basemodel.state_dict())
        model.linear_decoder.load_state_dict(encoder.out.state_dict())
        # TODO check encoder is properly garbage collected

        return 0

    print(f'Initializing model from {initial_checkpoint}')
    states = torch.load(initial_checkpoint)
    model.load_state_dict(states['model'])

    return 0


def main():
    args = parse_args()

    if args.log_wandb:
        wandb_init(
            output_dir=args.save_dir,
            project='histco-v2-mixer',
            name=os.path.basename(args.save_dir),
            resume='auto',
            config=args,
        )

    # Target-side tokenization
    formatter = blocky5()

    # Input-side tokenization
    tokenizer = load_tokenizer(
        model_domain='Multilingual_CANINE',
    )

    # Datasets
    dataset_train, dataset_val = setup_datasets(
        args=args,
        formatter=formatter,
        tokenizer=tokenizer,
    )

    # Data loaders
    data_loader_train = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        )
    data_loader_val = DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        )

    # Setup model, optimizer, scheduler
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = Seq2SeqMixerOccCANINE(
        model_domain='Multilingual_CANINE', # TODO make arg, discuss with Vedel
        num_classes=formatter.num_classes,
        num_classes_flat=1919,
        dropout_rate=args.dropout,
        decoder_dim_feedforward=args.decoder_dim_feedforward,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    total_steps = len(data_loader_train) * args.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps,
    )

    # Setup mixed loss
    loss_fn_seq2seq = BlockOrderInvariantLoss(
        pad_idx=PAD_IDX,
        nb_blocks=5,
        block_size=5,
    )
    loss_fn_linear = torch.nn.BCEWithLogitsLoss()
    loss_fn = LossMixer(
        loss_fn_seq2seq=loss_fn_seq2seq,
        loss_fn_linear=loss_fn_linear,
        seq2seq_weight=args.seq2seq_weight,
    ).to(device)

    # Load states
    current_step = load_states(
        save_dir=args.save_dir,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        initial_checkpoint=args.initial_checkpoint,
    )

    # Save arguments
    with open(os.path.join(args.save_dir, 'args.yaml'), 'w', encoding='utf-8') as args_file:
        args_file.write(
            yaml.safe_dump(args.__dict__, default_flow_style=False)
        )

    train(
        model=model,
        data_loaders={
            'data_loader_train': data_loader_train,
            'data_loader_val': data_loader_val,
        },
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        scheduler=scheduler,
        save_dir=args.save_dir,
        total_steps=total_steps,
        current_step=current_step,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        log_wandb=args.log_wandb,
    )


if __name__ == '__main__':
    main()
