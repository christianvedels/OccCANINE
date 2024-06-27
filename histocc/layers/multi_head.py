from torch import nn, Tensor


class SequenceEstimator(nn.Module):
    '''
    Creates a dense layer for each element in output_sizes with nodes defined
    by the sizes in output_sizes, then optionally applies a log_softmax to each
    layer.

    '''

    def __init__(
            self,
            input_size: int,
            num_classes: list[int],
            log_softmax: bool = False,
            ):
        super().__init__()

        for size in (input_size, *num_classes):
            self._validate_size(size)

        self.log_softmax = log_softmax
        self.fc_layers = nn.ModuleList(
            [nn.Linear(input_size, size) for size in num_classes]
        )

    def _validate_size(size: int):
        if not isinstance(size, int):
            raise TypeError(f'number of nodes must be integer but received {size} of type {type(size)}')

        if size <= 0:
            raise ValueError(f'number of ndoes must be positive but received {size}')

    def forward(self, x) -> tuple[Tensor, ...]:
        out = [fc(x) for fc in self.fc_layers]

        if self.log_softmax:
            out = [nn.functional.log_softmax(x, dim=1) for x in out]

        out = tuple(out)

        return out


class DenseSequenceEstimator(nn.Module):
    # TODO Consider more efficient dense version; however, this
    # forces the
    def __init__(self):
        super().__init__()


class MultiHeadClassifier(nn.Modue):
    def __init__(
            self,
            input_size: int,
            num_classes: list[int],
            drop_rate: float = None, # TODO consider dropping, as this is impl pre cls call
            ):
        super().__init__()

        self.dropout = nn.Dropout(drop_rate) if drop_rate else nn.Identity()
        self.classifier = SequenceEstimator(input_size, num_classes)

    def forward(self, x) -> tuple[Tensor, ...]:
        out = self.dropout(x)
        out = self.classifier(out)

        return out