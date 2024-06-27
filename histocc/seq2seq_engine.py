import time

import torch

from torch import nn, Tensor

from .formatter import PAD_IDX
from .utils import create_mask


class Averager:
    def __init__(self):
        self.sum: int | float = 0
        self.count: int = 0

    @property
    def avg(self) -> float:
        return self.sum / self.count

    def update(self, val: int | float, num: int = 1):
        self.sum += val * num
        self.count += num


def seq2seq_sequence_accuracy(
        output: Tensor,
        target: Tensor,
        pad_idx: int,
        ) -> tuple[Tensor, Tensor]:
    """
    Calculate the sequence and token accuracies of a seq2seq-style output and
    target. In both calculations, all indexes of `target` that correspond to
    padding are omitted from the calculations. Note that the token accuracy
    weighs longer sequences higher (proportional to length). However, as this
    function is likely called on batches (rather than entire data set), this
    will only be true in a batch-wise sense and the final token accuracy will
    weigh observations proportional to their length within batches but
    uniformly across batches.

    Parameters
    ----------
    output : Tensor
        [B, SeqLen, Features]-shaped model output.
    target : Tensor
        [B, SeqLen]-shaped targets.
    pad_idx : int
        Which index used to represent padding.

    Returns
    -------
    (Tensor, Tensor)
        1-element tensors with the sequence and token accuracy, respectively.

    """
    max_index = torch.argmax(output, dim=2).detach().cpu() # pylint: disable=E1101
    _target = target.clone() # To allow in-place operations to not change input

    # Padded elements should not be accounted for in calculation; set to -1 to
    # ensure they match, allowing the calculation of sequence accuracy used
    target_padding_mask = (_target == pad_idx)
    max_index[target_padding_mask] = -1
    _target[target_padding_mask] = -1

    is_eq = max_index == _target.detach().cpu()
    seq_acc = 100 * is_eq.all(axis=1).float().mean()

    # When calculating token accuracy, make sure not to inflate with the padded
    # values; hence remove those from both numerator and enumerator
    token_acc = 100 * (is_eq.float().sum() - target_padding_mask.sum()) / (~target_padding_mask).sum()

    return seq_acc, token_acc


def train_one_epoch(
        model: nn.Module,
        data_loader: torch.utils.data.DataLoader,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        log_interval: int = 100,
        ) -> tuple[float, float]:
    model = model.train()

    losses = Averager()
    batch_time = Averager()

    for batch_idx, batch in enumerate(data_loader):
        start = time.time()

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

        # Backward pass & step
        optimizer.zero_grad()
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        batch_time.update(time.time() - start)

        if batch_idx % log_interval == 0:
            print(f'Batch {batch_idx + 1} of {len(data_loader)}')
            print(f'Average loss: {losses.avg}')
            print(f'Average batch time: {batch_time.avg}')

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

        seq_acc, token_acc = seq2seq_sequence_accuracy( # FIXME some issue where seq acc > token acc
            outputs, targets[:, 1:], PAD_IDX,
        )
        seq_accs.update(seq_acc, outputs.size(0))
        token_accs.update(token_acc, outputs.size(0))

    return losses.avg, seq_accs.avg, token_accs.avg


def train(
        epochs: int,
        model: nn.Module,
        data_loaders: dict[str, torch.utils.data.DataLoader], # TODO split or use dataclass
        loss_fn: nn.Module,
        optimizer,
        device,
        scheduler,
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
