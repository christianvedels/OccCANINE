import torch

from torch import nn, Tensor

from .masking import generate_square_subsequent_mask
from ..model_assets import Seq2SeqOccCANINE, Seq2SeqMixerOccCANINE, CANINEOccupationClassifier
from ..formatter.hisco import BlockyHISCOFormatter
from .trie import TrieNode, build_trie # For full search


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
    prob_linear_topk, linear_topk = prob_linear_topk.detach(), linear_topk.detach()

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


def full_search_decoder_seq2seq_optimized(
        model: Seq2SeqOccCANINE,
        descr: torch.Tensor,
        input_attention_mask: torch.Tensor,
        device: torch.device,
        codes_list: list[list[int]],
        start_symbol: int,
        ) -> dict:

    memory = model.encode(descr, input_attention_mask)
    batch_size = descr.size(0)

    # Step 1: Build Trie
    trie = build_trie(codes_list)

    # Step 2: Initialize results
    results = torch.empty((batch_size, len(codes_list)), dtype=torch.float, device=device)
    code_indices = {tuple(code): idx for idx, code in enumerate(codes_list)}

    # Step 3: Initialize sequences
    seq = torch.ones(batch_size, 1).fill_(start_symbol).type(torch.long).to(device)
    prob_seq = torch.ones(batch_size, 1).fill_(1.0).type(torch.float).to(device)

    # Step 4: Decode using Trie
    stack = [(trie, seq, prob_seq)]

    # n_model_calls = 0

    while stack:
        node, seq, prob_seq = stack.pop()

        if node.codes:
            for code in node.codes:
                code_seq_probs = prob_seq[:, -1]
                results[:, code_indices[tuple(code)]] = code_seq_probs

        for number, child_node in node.children.items():
            which_output = torch.ones(batch_size, 1).fill_(number).type(torch.long).to(device)
            target_mask = generate_square_subsequent_mask(seq.shape[1], device).type(torch.bool)
            out = model.decode(
                memory=memory,
                target=seq,
                target_mask=target_mask,
                target_padding_mask=None,
            )[:, -1:, :]
            # n_model_calls += 1

            next_prob = torch.gather(torch.nn.functional.softmax(out, dim=2), 2, which_output.unsqueeze(2)).squeeze(2)
            new_prob_seq = prob_seq * next_prob
            new_seq = torch.cat([seq, which_output], dim=1)
            stack.append((child_node, new_seq, new_prob_seq))

    return results


def full_search_decoder_mixer_optimized(
        model: Seq2SeqMixerOccCANINE,
        descr: torch.Tensor,
        input_attention_mask: torch.Tensor,
        device: torch.device,
        codes_list: list[list[int]],
        start_symbol: int,
        ) -> dict:
    memory = model.encode(descr, input_attention_mask)
    
    # Ensure memory is a tensor
    if isinstance(memory, tuple):
        memory = memory[0]

    batch_size = descr.size(0)

    # Step 1: Build Trie
    trie = build_trie(codes_list)

    # Step 2: Initialize results
    results = torch.empty((batch_size, len(codes_list)), dtype=torch.float, device=device)
    code_indices = {tuple(code): idx for idx, code in enumerate(codes_list)}

    # Step 3: Initialize sequences
    seq = torch.ones(batch_size, 1).fill_(start_symbol).type(torch.long).to(device)
    prob_seq = torch.ones(batch_size, 1).fill_(1.0).type(torch.float).to(device)

    # Step 4: Decode using Trie
    stack = [(trie, seq, prob_seq)]

    # n_model_calls = 0

    while stack:
        node, seq, prob_seq = stack.pop()

        if node.codes:
            for code in node.codes:
                code_seq_probs = prob_seq[:, -1]
                results[:, code_indices[tuple(code)]] = code_seq_probs

        for number, child_node in node.children.items():
            which_output = torch.ones(batch_size, 1).fill_(number).type(torch.long).to(device)
            target_mask = generate_square_subsequent_mask(seq.shape[1], device).type(torch.bool)
            out = model.decode(
                memory=memory,
                target=seq,
                target_mask=target_mask,
                target_padding_mask=None,
            )[:, -1:, :]
            # n_model_calls += 1

            next_prob = torch.gather(torch.nn.functional.softmax(out, dim=2), 2, which_output.unsqueeze(2)).squeeze(2)
            new_prob_seq = prob_seq * next_prob
            new_seq = torch.cat([seq, which_output], dim=1)
            stack.append((child_node, new_seq, new_prob_seq))

    # print(n_model_calls) # 4073 is equal to trie.count_nodes(): Compared to len(codes_list)*5 = 9595

    return results


def _greedy_decode_with_banned_prefixes_mixer(
    model: Seq2SeqMixerOccCANINE,
    memory: torch.Tensor,
    start_symbol: int,
    code_len: int,
    device: torch.device,
    banned_codes_per_sample: list[list[list[int]]],  # [batch][num_banned][code_len]
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Greedy decode a fixed-length code while preventing reproduction of any banned code
    by masking next-token choices that would continue a banned prefix.

    Returns:
        seq: (batch, 1 + code_len) tensor of token ids (incl. BOS at position 0)
        prob_seq: (batch, 1 + code_len) tensor of step-wise probabilities (prob at BOS pos is 1.0)
    """
    batch_size = memory.size(0)
    seq = torch.full((batch_size, 1), start_symbol, dtype=torch.long, device=device)
    prob_seq = torch.ones((batch_size, 1), dtype=torch.float, device=device)

    for step in range(code_len):
        target_mask = generate_square_subsequent_mask(seq.shape[1], device).type(torch.bool)

        out = model.decode(
            memory=memory,
            target=seq,
            target_mask=target_mask,
            target_padding_mask=None,
        )[:, -1:, :]  # (batch, 1, vocab)

        # Copy logits for masking
        logits = out.clone()

        # Mask next-token options that would continue towards a banned code
        # If current prefix matches a banned code up to 'step', ban that banned token at position 'step'.
        for b in range(batch_size):
            banned_next: set[int] = set()
            prefix = seq[b, 1:].tolist()  # already generated tokens (no BOS), length == step
            for banned in banned_codes_per_sample[b]:
                if len(banned) > step and prefix == banned[:step]:
                    banned_next.add(int(banned[step]))
            if banned_next:
                logits[b, 0, list(banned_next)] = -1e9  # effectively remove these choices

        probs = torch.nn.functional.softmax(logits, dim=2)
        next_prob, next_token = torch.max(probs, dim=2)  # (batch, 1)

        seq = torch.cat([seq, next_token], dim=1)
        prob_seq = torch.cat([prob_seq, next_prob], dim=1)

    return seq, prob_seq


def top_k_decoder_mixer(
    model: Seq2SeqMixerOccCANINE,
    descr: torch.Tensor,
    input_attention_mask: torch.Tensor,
    device: torch.device,
    start_symbol: int,
    code_len: int,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Iterative greedy top-k for Seq2SeqMixer:
      - k times: greedy decode while banning previously found codes via prefix masks.
    No trie; avoids full search.

    Args:
        model: mixer model with .encode and .decode.
        descr: (batch, seq_len) input ids.
        input_attention_mask: (batch, seq_len) attention mask.
        device: torch device.
        start_symbol: BOS token id.
        code_len: number of tokens to generate (excluding BOS).
        k: number of hypotheses to return.

    Returns:
        codes_topk: (batch, k, code_len) int tensor.
        probs_topk: (batch, k) float tensor with product of step probabilities.
    """
    memory = model.encode(descr, input_attention_mask)
    if isinstance(memory, tuple):
        memory = memory[0]  # use seq2seq memory only

    batch_size = descr.size(0)
    k_eff = max(0, min(k,  max(1, k)))

    codes_topk = torch.zeros((batch_size, k_eff, code_len), dtype=torch.long, device=device)
    probs_topk = torch.zeros((batch_size, k_eff), dtype=torch.float, device=device)

    # Keep per-sample list of banned codes (as lists of ints of length code_len)
    banned_codes_per_sample: list[list[list[int]]] = [[] for _ in range(batch_size)]

    for i in range(k_eff):
        seq, prob_seq = _greedy_decode_with_banned_prefixes_mixer(
            model=model,
            memory=memory,
            start_symbol=start_symbol,
            code_len=code_len,
            device=device,
            banned_codes_per_sample=banned_codes_per_sample,
        )
        codes = seq[:, 1:1 + code_len]  # drop BOS
        probs = torch.prod(prob_seq[:, 1:], dim=1)  # product of step-wise probabilities

        codes_topk[:, i, :] = codes
        probs_topk[:, i] = probs

        # Add newly found codes to banned lists
        for b in range(batch_size):
            banned_codes_per_sample[b].append(codes[b].tolist())

    return codes_topk, probs_topk
