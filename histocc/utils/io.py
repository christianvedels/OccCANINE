import os

import torch

import pandas as pd

from transformers import CanineTokenizer

from histocc import (
    OccDatasetMixerInMemMultipleFiles,
    Seq2SeqMixerOccCANINE,
    CANINEOccupationClassifier_hub,
)
from histocc.formatter import (
    BlockyFormatter,
    EOS_IDX,
)


def load_states(
        save_dir: str,
        model: Seq2SeqMixerOccCANINE,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        initial_checkpoint: str | None = None,
        only_encoder: bool = False,
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
        model.linear_decoder.load_state_dict(encoder.out.state_dict()) # FIXME this leads to an issue when trying to set up model with other number of classes than that of the HISCO system
        # TODO check encoder is properly garbage collected

        return 0

    print(f'Initializing model from {initial_checkpoint}')
    states = torch.load(initial_checkpoint)

    if only_encoder:
        print('Only loading encoder from --initial-checkpoint')
        encoder_state_dict = {k[len("encoder."):]: v for k, v in states['model'].items() if k.startswith("encoder.")}
        model.encoder.load_state_dict(encoder_state_dict)
    else:
        model.load_state_dict(states['model'])

    return 0


def _check_if_data_prepared(save_path: str) -> dict[str, int] | None:
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


def _prepare_target_cols(
        data: pd.DataFrame,
        formatter: BlockyFormatter,
        drop_bad_rows: bool = False,
        allow_codes_shorter_than_block_size: bool = False,
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

    # Track whether length shorter that block size, as that may indicate
    # that leading zeros have been dropped. We can do this by tracking
    # whether there are any EOS_IDX present in formatted code ASIDE from
    # as its last element
    len_less_than_block_size: list[int] = []

    for i in range(len(data)):
        try:
            formatted = formatter.transform_label(data.iloc[i])
            passes_formatter.append(True)

            if EOS_IDX in formatted[:-1]:
                if data.iloc[i][formatter.target_cols[0]] == '?' and not EOS_IDX in formatted[(formatter.block_size + 1):-1]:
                    # OK to have EOS_IDX in FIRST code if due to missing -> '?' cast (see above)
                    pass
                else:
                    len_less_than_block_size.append(i)
        except: # pylint: disable=W0702
            passes_formatter.append(False)

        if (i + 1) % 10_000 == 0:
            print(f'Scanned {i + 1:,} of {len(data):,} observations.')

    bad_cases = len(passes_formatter) - sum(passes_formatter)

    if bad_cases > 0:
        if drop_bad_rows:
            print(f'Dropping {bad_cases} cases of labels not fit for formatter.')
            data = data[passes_formatter]
        else:
            raise ValueError(f'{bad_cases} bad cases of labels (of {len(data)}). If you to omit these rows, specify --drop-bad-labels')

    if len(len_less_than_block_size) > 0:
        if allow_codes_shorter_than_block_size:
            print(f'{len(len_less_than_block_size):,} cases of labels shorter than block size. Assuming such codes are allowed since --allow-codes-shorter-than-block-size was specified.')
        else:
            raise ValueError(f'{len(len_less_than_block_size):,} cases of labels shorter than block size, which may indicate that leading zeroes have been dropped by accident. If these cases are exptected, specify --allow-codes-shorter-than-block-size \nExample rows: {data.iloc[len_less_than_block_size].head(10)}')

    return data


def prepare_finetuning_data(
        dataset: str,
        input_col: str,
        formatter: BlockyFormatter,
        save_path: str,
        share_val: float,
        language: str = 'unk',
        language_col: str | None = None,
        drop_bad_rows: bool = False,
        allow_codes_shorter_than_block_size: bool = False,
) -> dict[str, int]:
    if not os.path.isdir(save_path):
        print(f'Creating fine-tuning directory {save_path}')
        os.makedirs(save_path, exist_ok=False)
    else:
        mapping = _check_if_data_prepared(save_path)

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
    data = _prepare_target_cols(
        data=data,
        formatter=formatter,
        drop_bad_rows=drop_bad_rows,
        allow_codes_shorter_than_block_size=allow_codes_shorter_than_block_size,
    )

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


def setup_finetuning_datasets(
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
