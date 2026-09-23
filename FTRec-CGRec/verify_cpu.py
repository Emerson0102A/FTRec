"""Small CPU probe of the published CGRec model and sample data.

This checks one optimization step and sampled ranking. Its metrics are diagnostics,
not a reproduction of the paper's full training or evaluation protocol.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))

from datasets import CausalDataset  # noqa: E402
from models import CausalModel  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-users", type=int, default=1)
    parser.add_argument("--eval-users", type=int, default=8)
    parser.add_argument("--max-seq-length", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    cli = parse_args()
    if min(cli.train_users, cli.eval_users, cli.max_seq_length, cli.hidden_size) <= 0:
        raise ValueError("all size arguments must be positive")
    if cli.hidden_size % 2:
        raise ValueError("hidden-size must be divisible by two attention heads")

    random.seed(cli.seed)
    np.random.seed(cli.seed)
    torch.manual_seed(cli.seed)
    torch.set_num_threads(2)

    # The downloaded sample contains only lists of strings; it was inspected with
    # pickletools before being added to this repository. Load only this fixed file.
    with (HERE / "src" / "amazon_list_dataset_202312_sampled.pkl").open("rb") as stream:
        user_sequences = pickle.load(stream)
    if len(user_sequences) < 2 or not user_sequences[0]:
        raise ValueError("CGRec sample has no user sequences")
    if max(cli.train_users, cli.eval_users) > len(user_sequences[0]):
        raise ValueError("requested more users than the CGRec sample contains")

    max_domain = max(int(value) for row in user_sequences[0] for value in row.split(","))
    max_item = max(int(value) for row in user_sequences[1] for value in row.split(","))
    config = SimpleNamespace(
        type_size=max_domain + 1,
        cat2_size=max_domain + 2,
        cat1_size=max_domain + 3,
        item_size=max_item + 1,
        hidden_size=cli.hidden_size,
        num_hidden_layers=2,
        num_attention_heads=2,
        hidden_act="gelu",
        attention_probs_dropout_prob=0.0,
        hidden_dropout_prob=0.0,
        initializer_range=0.02,
        max_seq_length=cli.max_seq_length,
        local_rank=0,
        loss_type="negative",
        hierarhical="y",
        shaply_value="y",
        except_type=[0, 1, 2, 3, 4],
    )
    model = CausalModel(config)
    model.device = torch.device("cpu")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train_data = CausalDataset(config, user_sequences, config.except_type, "train")
    eval_data = CausalDataset(config, user_sequences, config.except_type, "test")

    model.train()
    losses: list[float] = []
    for index in range(cli.train_users):
        batch = [tensor.unsqueeze(0) for tensor in train_data[index]]
        item_input, item_pos, item_neg, test_neg, item_answer = batch[:5]
        cat1_input, cat1_pos, cat1_neg = batch[5:8]
        cat2_input, cat2_pos, cat2_neg = batch[8:11]
        type_input = batch[11]
        loss, _, _, _ = model.pretrain_seq(
            item_input, item_pos, item_neg, test_neg, item_answer,
            cat1_input, cat1_pos, cat1_neg,
            cat2_input, cat2_pos, cat2_neg,
            type_input, config.hierarhical,
        )
        if not torch.isfinite(loss):
            raise ValueError("CGRec produced a non-finite training loss")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))

    model.eval()
    hits = 0
    ndcg = 0.0
    with torch.no_grad():
        for index in range(cli.eval_users):
            batch = [tensor.unsqueeze(0) for tensor in eval_data[index]]
            item_input, item_pos, item_neg, test_neg, item_answer = batch[:5]
            cat1_input = batch[5]
            cat2_input = batch[8]
            type_input = batch[11]
            _, _, recommendation = model.get_last_emb(
                item_input, cat1_input, cat2_input, type_input,
                item_pos, item_neg, config.hierarhical, cuda_yn="n",
            )
            candidates = torch.cat((item_answer, test_neg), dim=-1)
            scores = torch.sum(
                model.item_embeddings(candidates) * recommendation.unsqueeze(1),
                dim=-1,
            )[0]
            rank = int((scores[1:] > scores[0]).sum())
            if rank < 5:
                hits += 1
                ndcg += 1.0 / math.log2(rank + 2)

    print(json.dumps({
        "source": "cpark88/CGRec",
        "shaply_value": config.shaply_value,
        "train_users": cli.train_users,
        "eval_users": cli.eval_users,
        "train_loss": sum(losses) / len(losses),
        "hit_at_5": hits / cli.eval_users,
        "ndcg_at_5": ndcg / cli.eval_users,
        "candidate_count": 101,
        "device": "cpu",
    }, sort_keys=True))


if __name__ == "__main__":
    main()
