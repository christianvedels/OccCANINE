import os
import csv
import argparse
import hashlib

from collections import OrderedDict

try:
    import wandb
except ImportError:
    pass


def update_summary(step: int, metrics, filename, log_wandb=False):
    write_header = not os.path.isfile(filename)

    rowd = OrderedDict(step=step)
    rowd.update(metrics)

    with open(filename, mode='a') as summary:
        writer = csv.DictWriter(summary, fieldnames=rowd.keys())

        if write_header:
            writer.writeheader()

        writer.writerow(rowd)

    if log_wandb:
        wandb.log(rowd)


def pathhash(output_dir: str, target_len: int = 20):
    output_dir = os.path.normpath(output_dir).split(os.path.sep)
    output_dir = ''.join(output_dir)

    abbrev = hashlib.sha256(output_dir.encode()).hexdigest()

    if len(abbrev) < target_len:
        abbrev += '0' * (target_len - len(abbrev))
    elif len(abbrev) > target_len:
        abbrev = abbrev[:target_len]

    return abbrev


def wandb_init(
        output_dir: str,
        project: str,
        name: str,
        resume: str,
        config: argparse.Namespace,
        ):
    _dir = os.path.join('./wandb', pathhash(output_dir))

    if not os.path.exists(_dir):
        os.makedirs(_dir)

    wandb.init(
        project=project,
        name=name,
        dir=_dir,
        resume=resume,
        config=config,
        )
