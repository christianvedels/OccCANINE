import os
import time

import torch

from torch import nn
from sklearn.metrics import accuracy_score

from .formatter import PAD_IDX
from .utils import (
    create_mask,
    Averager,
    order_invariant_accuracy,
    update_summary,
)
from .model_assets import Seq2SeqMixerOccCANINE
from .loss import LossMixer


def train_one_epoch(
        model: Seq2SeqMixerOccCANINE,
        data_loader: torch.utils.data.DataLoader,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        current_step: int,
        log_interval: int = 100,
        eval_interval: int | None = None,
        save_interval: int | None = None,
        save_dir: str | None = None,
        data_loader_eval: torch.utils.data.DataLoader | None = None,
        log_wandb: bool = False,
        ) -> int:
    model = model.train()

    last_step = len(data_loader) - 1
    losses = Averager()
    batch_time = Averager()
    batch_time_data = Averager()
    samples_per_sec = Averager()

    # Need to initialize first "end time", as this is
    # calculated at bottom of batch loop
    end = time.time()

    for batch_idx, batch in enumerate(data_loader):
        current_step += 1

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        targets_seq2seq = batch['targets_seq2seq'].to(device)
        targets_linear = batch['targets_linear'].to(device)

        batch_time_data.update(time.time() - end)

        # Prepare target as input for seq2seq model
        target_seq2seq_input = targets_seq2seq[:, :-1]
        target_mask, target_padding_mask = create_mask(target_seq2seq_input, PAD_IDX, device)

        # Forward pass
        out_seq2seq, out_linear = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            target=target_seq2seq_input,
            target_mask=target_mask,
            target_padding_mask=target_padding_mask,
        )

        loss = loss_fn(
            out_seq2seq=out_seq2seq,
            out_linear=out_linear,
            target_seq2seq=targets_seq2seq,
            target_linear=targets_linear,
            )
        losses.update(loss.item(), out_seq2seq.size(0))

        # Backward pass & step
        optimizer.zero_grad()
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        elapsed = time.time() - end
        batch_time.update(elapsed)
        samples_per_sec.update(out_seq2seq.size(0) / elapsed)

        if batch_idx % log_interval == 0 or batch_idx == last_step:
            print(f'Batch {batch_idx + 1} of {len(data_loader)}. Batch time (data): {batch_time.avg:.2f} ({batch_time_data.avg:.2f}). Train loss: {losses.avg:.6f}')
            # print(f'Samples/second: {samples_per_sec.avg:.2f}')
            # print(f'Max. memory allocated/reserved: {torch.cuda.max_memory_allocated() / (1024 ** 3):.2f}/{torch.cuda.max_memory_reserved() / (1024 ** 3):.2f} GB')

        if save_interval is not None and current_step % save_interval == 0:
            states = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'step': current_step,
                'key': data_loader.dataset.map_code_label,
            }
            torch.save(states, os.path.join(save_dir, f'{current_step}.bin'))
            torch.save(states, os.path.join(save_dir, 'last.bin'))

        if eval_interval is not None and current_step % eval_interval == 0:
            print('Starting eval pass')
            eval_loss, eval_loss_linear, eval_loss_seq2seq, eval_seq_acc, eval_token_acc, eval_flat_acc = evaluate(
                model=model,
                data_loader=data_loader_eval,
                loss_fn=loss_fn,
                device=device,
            )
            model.train()

            update_summary(
                current_step,
                metrics={
                    'batch_time': batch_time.avg,
                    'batch_time_data': batch_time_data.avg,
                    'train_loss': losses.avg,
                    'val_loss': eval_loss,
                    'val_loss_linear': eval_loss_linear,
                    'val_loss_seq2seq': eval_loss_seq2seq,
                    'seq_acc': eval_seq_acc,
                    'token_acc': eval_token_acc,
                    'flat_acc': eval_flat_acc,
                    'lr': scheduler.get_last_lr()[0],
                },
                filename=os.path.join(save_dir, 'logs.csv'),
                log_wandb=log_wandb,
            )

        end = time.time()

    return current_step


@torch.no_grad
def evaluate(
        model: nn.Module,
        data_loader: torch.utils.data.DataLoader,
        loss_fn: nn.Module,
        device: torch.device,
        log_interval: int = 100,
        ):
    model = model.eval()

    losses = Averager()
    losses_linear = Averager()
    losses_seq2seq = Averager()

    token_accs = Averager()
    seq_accs = Averager()
    flat_accs = Averager()

    for batch_idx, batch in enumerate(data_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        targets_seq2seq = batch['targets_seq2seq'].to(device)
        targets_linear = batch['targets_linear'].to(device)

        # Prepare target as input for seq2seq model
        target_seq2seq_input = targets_seq2seq[:, :-1]
        target_mask, target_padding_mask = create_mask(target_seq2seq_input, PAD_IDX, device)

        # Forward pass
        out_seq2seq, out_linear = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            target=target_seq2seq_input,
            target_mask=target_mask,
            target_padding_mask=target_padding_mask,
        )

        loss = loss_fn(
            out_seq2seq=out_seq2seq,
            out_linear=out_linear,
            target_seq2seq=targets_seq2seq,
            target_linear=targets_linear,
            )
        loss_linear = loss_fn.loss_fn_linear(out_linear, targets_linear)
        loss_seq2seq = loss_fn.loss_fn_seq2seq(out_seq2seq, targets_seq2seq)

        losses.update(loss.item(), out_seq2seq.size(0))
        losses_linear.update(loss_linear.item(), out_seq2seq.size(0))
        losses_seq2seq.update(loss_seq2seq.item(), out_seq2seq.size(0))

        seq_acc, token_acc = order_invariant_accuracy(
            output=out_seq2seq,
            target=targets_seq2seq[:, 1:],
            pad_idx=PAD_IDX,
            nb_blocks=loss_fn.loss_fn_seq2seq.nb_blocks,
            block_size=loss_fn.loss_fn_seq2seq.block_size,
        )
        seq_accs.update(seq_acc.item(), out_seq2seq.size(0))
        token_accs.update(token_acc.item(), out_seq2seq.size(0))

        # Linear decoder accuracy
        preds_linear = torch.sigmoid(out_linear) > 0.5
        preds_linear = preds_linear.float().cpu()

        acc_flat = accuracy_score(preds_linear, targets_linear.cpu())
        flat_accs.update(acc_flat, preds_linear.size(0))

        if batch_idx % log_interval == 0:
            print(f'Batch {batch_idx + 1} of {len(data_loader)}. Accuracy (seq/token/flat): ({seq_accs.avg:.2f}/{token_accs.avg:.2f}/{flat_accs.avg:.2f}). Validation loss: {losses.avg:.6f}')

    return losses.avg, losses_linear.avg, losses_seq2seq.avg, seq_accs.avg, token_accs.avg, flat_accs.avg


def train(
        model: Seq2SeqMixerOccCANINE,
        data_loaders: dict[str, torch.utils.data.DataLoader], # TODO split or use dataclass
        loss_fn: LossMixer,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        save_dir: str,
        total_steps: int,
        current_step: int = 0,
        log_interval: int = 100,
        eval_interval: int = 1000,
        save_interval: int = 1000,
        log_wandb: bool = False,
        ):
    while current_step < total_steps:
        print(f'Completed {current_step} of {total_steps} steps. Starting new epoch.')

        current_step = train_one_epoch(
            model,
            data_loaders['data_loader_train'],
            loss_fn,
            optimizer,
            device,
            scheduler,
            current_step=current_step,
            log_interval=log_interval,
            eval_interval=eval_interval,
            save_interval=save_interval,
            save_dir=save_dir,
            data_loader_eval=data_loaders['data_loader_val'],
            log_wandb=log_wandb,
        )
