import torch

from torch import nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from .model_assets import Seq2SeqOccCANINE
from .prediction_assets import OccCANINE, get_adapated_tokenizer
from .seq2seq_engine import train
from .loss import Seq2SeqCrossEntropy
from .formatter import PAD_IDX


class Seq2SeqOccCANINEWrapper(OccCANINE):
    def __init__( # pylint: disable=W0231
            self,
            name: str = "CANINE",
            device: str | None = None,
            batch_size: int = 256,
            verbose: bool = False,
            hf: bool = True,
    ):
        # Detect device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else torch.device(device)
        if verbose:
            print(f"Using device: {self.device}")

        if name == "CANINE" and not hf:
            raise ValueError("When 'hf' is False, a specific local model 'name' must be provided.")

        self.name = name
        self.batch_size = batch_size
        self.verbose = verbose

        # Get tokenizer
        self.tokenizer = get_adapated_tokenizer("CANINE_Multilingual_CANINE_sample_size_10_lr_2e-05_batch_size_256")

        # Get key
        self.key, self.key_desc = self._load_keys()
        self.model = self.init_seq2seq_model()

        # Promise for later initialization
        self.finetune_key = None

    def init_seq2seq_model(self) -> nn.Module:
        model = Seq2SeqOccCANINE(
            model_domain='Multilingual_CANINE', # TODO make arg
            num_classes=[18] * 27, # TODO make arg, extract from formatter
        ).to(self.device)

        return model

    def _train_model(
            self,
            processed_data,
            model_name,
            epochs,
            only_train_final_layer,
            verbose = True,
            verbose_extra = False,
            new_labels = False,
            save_model = True,
            save_path = '../OccCANINE/Finetuned/',
            ):
        optimizer = AdamW(self.model.parameters(), lr=2*10**-5)
        total_steps = len(processed_data['data_loader_train']) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps,
        )

        # Set the loss function
        loss_fn = Seq2SeqCrossEntropy(pad_idx=PAD_IDX).to(self.device)

        train(
            epochs=epochs,
            model=self.model,
            data_loaders=processed_data,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=self.device,
            scheduler=scheduler,
            )
