import argparse
import os

import torch
import yaml

from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    get_linear_schedule_with_warmup,
    CanineTokenizer,
)

import pandas as pd

from histocc import (
    OccDatasetMixerInMemMultipleFiles,
    load_tokenizer,
    Seq2SeqMixerOccCANINE,
    BlockOrderInvariantLoss,
    LossMixer,
)
from histocc.seq2seq_mixer_engine import train
from histocc.formatter import (
    BlockyFormatter,
    construct_general_purpose_formatter,
    PAD_IDX,
)
from histocc.utils import wandb_init

try:
    # want to do import to set has_wandb even if not used directly
    import wandb # pylint: disable=W0611
    has_wandb = True # pylint: disable=C0103
except ImportError:
    has_wandb = False # pylint: disable=C0103

from train_mixer import load_states


def parse_args():
    parser = argparse.ArgumentParser()

    # File paths, data & model choices
    parser.add_argument('--save-path', type=str, default='./Finetuned/', help='Directory to store fine-tuned model, incl. processed data')
    parser.add_argument('--save-interval', type=int, default=1000, help='Number of steps between saving model')
    parser.add_argument('--dataset', type=str, default=None, help='Filename of dataset')
    parser.add_argument('--input-col', type=str, default='occ1', help='Column name of column with occupational descriptions')
    parser.add_argument('--target-cols', type=str, nargs='+', default=None, help='List of column names with labels')

    # Language (must specify none or one of below, cannot specify both)
    parser.add_argument('--language', type=str, default='unk', help='Occupational description language')
    parser.add_argument('--language-col', type=str, default=None, help='Optional column name in --dataset with language')

    # Data settings
    parser.add_argument('--block-size', type=int, default=5, help='Maximum number of characters in target (e.g., this is 5 for the HISCO system)')
    parser.add_argument('--share-val', type=float, default=0.1, help='Share of data set aside for tracking model performance')
    parser.add_argument('--use-within-block-sep', action='store_true', default=False, help='Whether to use "," as a separator for tokens WITHIN a code. Useful for, e.g., PSTI')
    parser.add_argument('--drop-bad-labels', action='store_true', default=False, help='Omit all observations where labels not adhere to formatting rules.')

    # Logging parameters
    parser.add_argument('--log-interval', type=int, default=100, help='Number of steps between reporting training stats')
    parser.add_argument('--eval-interval', type=int, default=1000, help='Number of steps between calculating and logging validation performance')
    parser.add_argument('--log-wandb', action='store_true', default=False, help='Whether to log validation performance using W&B')
    parser.add_argument('--wandb-project-name', type=str, default='histco-v2-mixer')

    # Data parameters
    parser.add_argument('--num-epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=128)

    # Model and optimizer parameters
    parser.add_argument('--learning-rate', type=float, default=2e-05)
    parser.add_argument('--seq2seq-weight', type=float, default=0.1)
    parser.add_argument('--warmup-steps', type=int, default=0)

    # Model initialization
    parser.add_argument('--initial-checkpoint', type=str, default=None, help='Model weights to use for initialization. Discarded if resume state exists at --save-path')
    parser.add_argument('--only-encoder', action='store_true', default=False, help='Only attempt to load encoder part of --initial-checkpoint')

    # Freezing
    parser.add_argument('--freeze-encoder', action='store_true', default=False)

    args = parser.parse_args()

    if args.language != 'unk' and args.language_col is not None:
        raise ValueError('Only specify one of --language and --language-col')

    if args.log_wandb and not has_wandb:
        raise ImportError('Specified --log-wandb, but wandb is not installed')

    return args


def check_if_data_prepared(save_path: str) -> dict[str, int] | None:
    if not os.path.isfile(os.path.join(save_path, 'data_train.csv')):
        return None

    if not os.path.isfile(os.path.join(save_path, 'data_val.csv')):
        return None

    if not os.path.isfile(os.path.join(save_path, 'key.csv')):
        return None

    mapping_df = pd.read_csv(
        os.path.join(save_path, 'key.csv'),
        dtype={'system_code': str, 'code': int},
        )
    mapping = dict(mapping_df.values)

    return mapping


def prepare_target_cols(
        data: pd.DataFrame,
        formatter: BlockyFormatter,
        drop_bad_rows: bool = False,
) -> pd.DataFrame:
    # All cases of space (' ') are cast to NaN
    for i, target_col in enumerate(formatter.target_cols):
        # Some NaN values instead coded as spaces
        data[target_col] = data[target_col].replace(' ', None)

    # First colummn should not contain any NaN -> use the '?' token instead
    assert '?' in formatter.map_char_idx
    data[formatter.target_cols[0]] = data[formatter.target_cols[0]].fillna('?')

    # Send all through formatter and track whether that works
    passes_formatter: list[bool] = []

    for i in range(len(data)):
        try:
            _ = formatter.transform_label(data.iloc[i])
            passes_formatter.append(True)
        except: # pylint: disable=W0702
            passes_formatter.append(False)

        if (i + 1) % 10_000 == 0:
            print(f'Scanned {i + 1} of {len(data)} observations.')

    bad_cases = len(passes_formatter) - sum(passes_formatter)

    if bad_cases > 0:
        if drop_bad_rows:
            print(f'Dropping {bad_cases} cases of labels not fit for formatter.')
            data = data[passes_formatter]
        else:
            raise ValueError(f'{bad_cases} bad cases of labels (of {len(data)}). If you to omit these rows, specify --drop-bad-labels')

    return data


def prepare_data(
        dataset: str,
        input_col: str,
        formatter: BlockyFormatter,
        save_path: str,
        share_val: float,
        language: str = 'unk',
        language_col: str | None = None,
        drop_bad_rows: bool = False,
) -> dict[str, int]:
    if not os.path.isdir(save_path):
        print(f'Creating fine-tuning directory {save_path}')
        os.makedirs(save_path, exist_ok=False)
    else:
        mapping = check_if_data_prepared(save_path)

        if mapping is not None:
            print(f'Prepared data exists at {save_path}, using that')
            return mapping

    # Load
    data: pd.DataFrame = pd.read_csv(dataset, dtype=str)

    # Select columns
    if language_col is None:
        data['lang'] = language
    else:
        data['lang'] = data[language_col]

    data = data[[input_col, *formatter.target_cols, 'lang']]
    data = data.rename(columns={input_col: 'occ1'})

    # Value checks, subsetting, and changing some values
    data = prepare_target_cols(data, formatter, drop_bad_rows)

    # Build code <-> label mapping
    unique_values = pd.unique(data[formatter.target_cols].values.ravel())
    unique_values = [val for val in unique_values if val is not None]

    mapping: dict[str, int] = {
        code: i for i, code in enumerate(unique_values)
        }
    mapping_df = pd.DataFrame(mapping.items(), columns=['system_code', 'code'])

    # Split data into train and validation
    data_val = data.sample(int(len(data) * share_val), random_state=42)
    data_train = data.drop(data_val.index)

    # Save datasets & mapping in specified fine-tuning folder
    data_train.to_csv(os.path.join(save_path, 'data_train.csv'), index=False)
    data_val.to_csv(os.path.join(save_path, 'data_val.csv'), index=False)
    mapping_df.to_csv(os.path.join(save_path, 'key.csv'), index=False)

    return mapping


def setup_datasets(
        target_cols: list[str],
        save_path: str,
        formatter: BlockyFormatter,
        tokenizer: CanineTokenizer,
        num_classes_flat: int,
        map_code_label: dict[str, int],
) -> tuple[OccDatasetMixerInMemMultipleFiles, OccDatasetMixerInMemMultipleFiles]:
    dataset_train = OccDatasetMixerInMemMultipleFiles(
        fnames_data=[os.path.join(save_path, 'data_train.csv')],
        formatter=formatter,
        tokenizer=tokenizer,
        max_input_len=128,
        num_classes_flat=num_classes_flat,
        training=True,
        target_cols=target_cols,
        map_code_label=map_code_label,
    )

    dataset_val = OccDatasetMixerInMemMultipleFiles(
        fnames_data=[os.path.join(save_path, 'data_val.csv')],
        formatter=formatter,
        tokenizer=tokenizer,
        max_input_len=128,
        num_classes_flat=num_classes_flat,
        training=False,
        target_cols=target_cols,
        map_code_label=map_code_label,
    )

    return dataset_train, dataset_val


def main():
    # Arguments
    args = parse_args()

    # Target-side tokenization
    formatter = construct_general_purpose_formatter(
        block_size=args.block_size,
        target_cols=args.target_cols,
        use_within_block_sep=args.use_within_block_sep,
    )

    # Input-side tokenization
    tokenizer = load_tokenizer(
        model_domain='Multilingual_CANINE',
    )

    # Data prep
    map_code_label = prepare_data(
        dataset=args.dataset,
        input_col=args.input_col,
        formatter=formatter,
        save_path=args.save_path,
        share_val=args.share_val,
        language=args.language,
        language_col=args.language_col,
        drop_bad_rows=args.drop_bad_labels,
    )
    num_classes_flat = len(map_code_label)

    if args.log_wandb:
        wandb_init(
            output_dir=args.save_path,
            project=args.wandb_project_name,
            name=os.path.basename(args.save_path),
            resume='auto',
            config=args,
        )

    # Load datasets
    dataset_train, dataset_val = setup_datasets(
        target_cols=args.target_cols,
        save_path=args.save_path,
        formatter=formatter,
        tokenizer=tokenizer,
        num_classes_flat=num_classes_flat,
        map_code_label=map_code_label,
    )

    # Data loaders
    data_loader_train = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        )
    data_loader_val = DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        shuffle=False,
        )

    # Setup model, optimizer, scheduler
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = Seq2SeqMixerOccCANINE(
        model_domain='Multilingual_CANINE',
        num_classes=formatter.num_classes,
        num_classes_flat=num_classes_flat,
    ).to(device)

    if args.freeze_encoder:
        for param in model.encoder.parameters():
            param.requires_grad = False

        optimizer = AdamW([param for name, param in model.named_parameters() if not name.startswith("encoder.")], lr=args.learning_rate)
    else:
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
        nb_blocks=formatter.max_num_codes,
        block_size=formatter.block_size,
    )
    loss_fn_linear = torch.nn.BCEWithLogitsLoss()
    loss_fn = LossMixer(
        loss_fn_seq2seq=loss_fn_seq2seq,
        loss_fn_linear=loss_fn_linear,
        seq2seq_weight=args.seq2seq_weight,
    ).to(device)

    # Load states
    current_step = load_states(
        save_dir=args.save_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        initial_checkpoint=args.initial_checkpoint,
        only_encoder=args.only_encoder,
    )

    # Save arguments
    with open(os.path.join(args.save_path, 'args.yaml'), 'w', encoding='utf-8') as args_file:
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
        save_dir=args.save_path,
        total_steps=total_steps,
        current_step=current_step,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        log_wandb=args.log_wandb,
        save_interval=args.save_interval
    )


if __name__ == '__main__':
    main()
