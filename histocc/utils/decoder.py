import torch

from torch import nn, Tensor

from .masking import generate_square_subsequent_mask
from ..model_assets import Seq2SeqOccCANINE, Seq2SeqMixerOccCANINE, CANINEOccupationClassifier
from ..formatter.hisco import BlockyHISCOFormatter


def greedy_decode(
        model: Seq2SeqOccCANINE,
        descr: Tensor,
        input_attention_mask: Tensor,
        device: torch.device,
        max_len: int,
        start_symbol: int,
        ) -> tuple[Tensor, Tensor]:
    memory = model.encode(descr, input_attention_mask)
    batch_size = descr.size(0)

    # Initialize sequence by placing BoS symbol.
    seq = torch.ones(batch_size, 1).fill_(start_symbol).type(torch.long).to(device)
    prob_seq = torch.ones(batch_size, 1).fill_(1.0).type(torch.long).to(device)

    for _ in range(max_len - 1):
        target_mask = generate_square_subsequent_mask(seq.shape[1], device).type(torch.bool) # TODO do we need cast?

        out = model.decode(
            memory=memory,
            target=seq,
            target_mask=target_mask,
            target_padding_mask=None,
            )[:, -1:, :] # Only use the prediction for the next token in seq

        next_token = torch.argmax(out, dim=2).detach()
        next_prob = torch.max(nn.functional.softmax(out, dim=2), dim=2)[0].detach()

        # Extend sequence by adding prediction of next token.
        seq = torch.cat([seq, next_token], dim=1)
        prob_seq = torch.cat([prob_seq, next_prob], dim=1)

    return seq, prob_seq


def mixer_greedy_decode(
        model: Seq2SeqMixerOccCANINE,
        descr: Tensor,
        input_attention_mask: Tensor,
        device: torch.device,
        max_len: int,
        start_symbol: int,
        linear_topk: int = 5,
        ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    memory, pooled_memory = model.encode(descr, input_attention_mask)
    batch_size = descr.size(0)

    # Linear output
    out_linear = model.linear_decoder(pooled_memory)
    out_linear = model.linear_decoder_drop(out_linear)
    prob_linear_topk, linear_topk = torch.sigmoid(out_linear).topk(linear_topk, axis=1)

    # seq2seq output
    # Initialize sequence by placing BoS symbol.
    seq = torch.ones(batch_size, 1).fill_(start_symbol).type(torch.long).to(device)
    prob_seq = torch.ones(batch_size, 1).fill_(1.0).type(torch.long).to(device)

    for _ in range(max_len - 1):
        target_mask = generate_square_subsequent_mask(seq.shape[1], device).type(torch.bool) # TODO do we need cast?

        out = model.decode(
            memory=memory,
            target=seq,
            target_mask=target_mask,
            target_padding_mask=None,
            )[:, -1:, :] # Only use the prediction for the next token in seq

        next_token = torch.argmax(out, dim=2).detach()
        next_prob = torch.max(nn.functional.softmax(out, dim=2), dim=2)[0].detach()

        # Extend sequence by adding prediction of next token.
        seq = torch.cat([seq, next_token], dim=1)
        prob_seq = torch.cat([prob_seq, next_prob], dim=1)

    return seq, prob_seq, linear_topk, prob_linear_topk

def flat_decode_flat_model(
        model: Seq2SeqMixerOccCANINE,
        descr: Tensor,
        input_attention_mask: Tensor,
        ):
    """
    Minimal decoder to handle everything as decoders in same module.
    Flat decoder for decoding based on 'flat' model (v1 of OccCANINE).
    
    """
    
    logits = model.forward(descr, input_attention_mask)
    
    return logits

def flat_decode_mixer(
        model: Seq2SeqMixerOccCANINE,
        descr: Tensor,
        input_attention_mask: Tensor,
        ):
    """
    Minimal decoder used for fast 'flat' decoding of mixed output models. 
    
    """
    _, pooled_memory = model.encode(descr, input_attention_mask)

    # Linear output
    logits = model.linear_decoder(pooled_memory)
            
    return logits


def greedy_decode_for_training(
        model: Seq2SeqOccCANINE,
        descr: Tensor,
        input_attention_mask: Tensor,
        device: torch.device,
        max_len: int,
        start_symbol: int,
        ) -> tuple[Tensor, Tensor]:
    memory = model.encode(descr, input_attention_mask)
    batch_size = descr.size(0)

    # Initialize sequence by placing BoS symbol.
    seq = torch.ones(batch_size, 1).fill_(start_symbol).type(torch.long).to(device).detach()
    output_seq = []

    for _ in range(max_len): # we loop all the way to fill in some value at EOS pos
        target_mask = generate_square_subsequent_mask(seq.shape[1], device).type(torch.bool) # TODO do we need cast?

        out = model.decode(
            memory=memory,
            target=seq,
            target_mask=target_mask,
            target_padding_mask=None,
            )[:, -1:, :] # Only use the prediction for the next token in seq

        next_token = torch.argmax(out, dim=2).detach()

        # Extend sequence by adding prediction of next token.
        seq = torch.cat([seq, next_token], dim=1)
        output_seq.append(out)

    output_seq = torch.cat(output_seq, dim=1)

    return output_seq

def decode_specific_code_seq2seq(
        model: Seq2SeqOccCANINE,
        descr: Tensor,
        input_attention_mask: Tensor,
        device: torch.device,
        code: list[int],
        start_symbol: int,
        formatter: BlockyHISCOFormatter
        ) -> float:
    memory = model.encode(descr, input_attention_mask)
    batch_size = descr.size(0)

    # Initialize sequence by placing BoS symbol.
    seq = torch.ones(batch_size, 1).fill_(start_symbol).type(torch.long).to(device)
    prob_seq = torch.ones(batch_size, 1).fill_(1.0).type(torch.long).to(device)

    for i in range(len(code)):
        target_mask = generate_square_subsequent_mask(seq.shape[1], device).type(torch.bool)

        out = model.decode(
            memory=memory,
            target=seq,
            target_mask=target_mask,
            target_padding_mask=None,
        )[:, -1:, :]
        
        next_token = torch.argmax(out, dim=2).detach()
        next_prob = torch.max(nn.functional.softmax(out, dim=2), dim=2)[0].detach()

        # Extend sequence by adding prediction of next token.
        seq = torch.cat([seq, next_token], dim=1)
        prob_seq = torch.cat([prob_seq, next_prob], dim=1)

        if i < len(code) - 1:
            # Set the next token in the sequence to the desired code value
            seq[:, -1] = code[i + 1]

    # Find the probability of the specific code sequence
    code_seq_prob = prob_seq[0, -1].item()

    return code_seq_prob
