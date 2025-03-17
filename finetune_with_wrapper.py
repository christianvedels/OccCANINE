import argparse

from histocc import OccCANINE


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
    parser.add_argument('--share-val', type=float, default=0.1, help='Share of data set aside for tracking model performance')
    parser.add_argument('--drop-bad-labels', action='store_true', default=False, help='Omit all observations where labels not adhere to formatting rules.')
    parser.add_argument('--allow-codes-shorter-than-block-size', action='store_true', default=False, help='Allow for codes shorter than block size. If not specified, such codes will raise an error, as they may indicate that leading zeroes have accidently been dropped.')
    parser.add_argument('--use-within-block-sep', action='store_true', default=False, help='Whether to use "," as a separator for tokens WITHIN a code. Useful for, e.g., PSTI')

    # Logging parameters
    parser.add_argument('--log-interval', type=int, default=100, help='Number of steps between reporting training stats')
    parser.add_argument('--eval-interval', type=int, default=1000, help='Number of steps between calculating and logging validation performance')

    # Data parameters
    parser.add_argument('--num-epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=128)

    # Model and optimizer parameters
    parser.add_argument('--learning-rate', type=float, default=2e-05)
    parser.add_argument('--seq2seq-weight', type=float, default=0.1)
    parser.add_argument('--warmup-steps', type=int, default=0)

    # Model initialization
    parser.add_argument('--initial-checkpoint', type=str, default=None, help='Model weights to use for initialization. Discarded if resume state exists at --save-path')
    parser.add_argument('--only-encoder', action='store_true', default=False, help='Only attempt to load encoder part of --initial-checkpoint') # FIXME this is currently ignored

    # Freezing
    parser.add_argument('--freeze-encoder', action='store_true', default=False)

    args = parser.parse_args()

    if args.language != 'unk' and args.language_col is not None:
        raise ValueError('Only specify one of --language and --language-col')

    return args


def main():
    # Arguments
    args = parse_args()

    wrapper = OccCANINE(
        batch_size=args.batch_size,
        target_cols=args.target_cols,
        use_within_block_sep=args.use_within_block_sep,
        name=args.initial_checkpoint,
        system='',
        hf=False,
    )

    wrapper.finetune(
        dataset=args.dataset,
        save_path=args.save_path,
        input_col=args.input_col,
        language=args.language,
        language_col=args.language_col,
        save_interval=args.save_interval,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        drop_bad_labels=args.drop_bad_labels,
        allow_codes_shorter_than_block_size=args.allow_codes_shorter_than_block_size,
        share_val=args.share_val,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        warmup_steps=args.warmup_steps,
        seq2seq_weight=args.seq2seq_weight,
        freeze_encoder=args.freeze_encoder,
    )


if __name__ == '__main__':
    main()
