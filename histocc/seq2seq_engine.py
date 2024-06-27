import time

import torch

from torch import nn, Tensor

from .formatter import PAD_IDX
from .utils import create_mask, Averager, seq2seq_sequence_accuracy

from .utils.metrics import order_invariant_accuracy


def train_one_epoch(
        model: nn.Module,
        data_loader: torch.utils.data.DataLoader,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        log_interval: int = 100,
        eval_and_save_interval: int | None = None,
        ) -> tuple[float, float]:
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
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        targets = batch['targets'].to(device)

        batch_time_data.update(time.time() - end)

        # Prepare target as input for seq2seq model
        target_input = targets[:, :-1]
        target_mask, target_padding_mask = create_mask(target_input, PAD_IDX, device)

        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            target=target_input,
            target_mask=target_mask,
            target_padding_mask=target_padding_mask,
        )

        loss = loss_fn(outputs, targets)
        losses.update(loss.item(), outputs.size(0))

        # Backward pass & step
        optimizer.zero_grad()
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        elapsed = time.time() - end
        batch_time.update(elapsed)
        samples_per_sec.update(outputs.size(0) / elapsed)

        if batch_idx % log_interval == 0 or batch_idx == last_step:
            print(f'Batch {batch_idx + 1} of {len(data_loader)}')
            print(f'Average loss: {losses.avg:.2f}')
            print(f'Average batch time: {batch_time.avg:.2f}')
            print(f'Average data batch time: {batch_time_data.avg:.2f}')
            print(f'Samples/second: {samples_per_sec.avg:.2f}')

            print(f'Max. memory allocated/reserved: {torch.cuda.max_memory_allocated() / (1024 ** 3):.2f}/{torch.cuda.max_memory_reserved() / (1024 ** 3):.2f} GB')

        if eval_and_save_interval is not None and (batch_idx + 1) % eval_and_save_interval == 0:
            # FIXME to properly implement this, we need to pass in val dataloader, save path(s), etc
            torch.save(
                model.state_dict(),
                f'Y:/pc-to-Y/tmp/order-invariant/occ-canine-{batch_idx}.bin',
            )
            # TODO also need to save optimizer and scheduler states

        end = time.time()

    return losses.avg, batch_time.avg


@torch.no_grad
def evaluate(
        model: nn.Module,
        data_loader: torch.utils.data.DataLoader,
        loss_fn: nn.Module,
        device: torch.device,
        ):
    model = model.eval()

    losses = Averager()
    token_accs = Averager()
    seq_accs = Averager()

    for batch in data_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        targets = batch['targets'].to(device)

        # Prepare target as input for seq2seq model
        target_input = targets[:, :-1]
        target_mask, target_padding_mask = create_mask(target_input, PAD_IDX, device)

        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            target=target_input,
            target_mask=target_mask,
            target_padding_mask=target_padding_mask,
        )

        loss = loss_fn(outputs, targets)
        losses.update(loss.item(), outputs.size(0))

        seq_acc, token_acc = order_invariant_accuracy(
            outputs, targets[:, 1:], PAD_IDX, 5, 5,
        )
        seq_accs.update(seq_acc, outputs.size(0))
        token_accs.update(token_acc, outputs.size(0))

        # seq_acc, token_acc = seq2seq_sequence_accuracy(
        #     outputs, targets[:, 1:], PAD_IDX,
        # )
        # seq_accs.update(seq_acc, outputs.size(0))
        # token_accs.update(token_acc, outputs.size(0))

    return losses.avg, seq_accs.avg, token_accs.avg


def train(
        epochs: int,
        model: nn.Module,
        data_loaders: dict[str, torch.utils.data.DataLoader], # TODO split or use dataclass
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        ):
    for epoch in range(1, epochs + 1):
        print(f'Epoch {epoch} of {epochs}')

        avg_train_loss, avg_batch_time = train_one_epoch(
            model,
            data_loaders['data_loader_train'],
            loss_fn,
            optimizer,
            device,
            scheduler,
            eval_and_save_interval=1000, # FIXME pass as arg
        )

        # Evaluate
        val_loss, seq_acc, token_acc = evaluate(
            model,
            data_loaders['data_loader_val'],
            loss_fn,
            device,
            )

        print(f'Validation loss: {val_loss}')
        print(f'Validation sequence accuracy: {seq_acc}')
        print(f'Validation token accuracy: {token_acc}')
