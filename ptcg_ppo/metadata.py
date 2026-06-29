from __future__ import annotations
import torch
from cg.api import all_attack, all_card_data


def cg_metadata_tensors(card_vocab_size: int = 4096, attack_vocab_size: int = 4096, max_attack_cost: int = 8):
    """Build the immutable metadata tables consumed by ObservationEncoder."""
    card_features = torch.zeros((card_vocab_size + 1, 14), dtype=torch.long)
    attack_features = torch.zeros((attack_vocab_size + 1, 6), dtype=torch.long)
    attack_cost_energy_types = torch.full((attack_vocab_size + 1, max_attack_cost), -1, dtype=torch.long)
    attack_cost_counts = torch.zeros((attack_vocab_size + 1,), dtype=torch.long)

    for card in all_card_data():
        cid = int(card.cardId)
        if not 0 <= cid <= card_vocab_size:
            continue
        card_features[cid] = torch.tensor([
            1, int(card.cardType), int(card.hp), int(card.retreatCost),
            -1 if card.weakness is None else int(card.weakness),
            -1 if card.resistance is None else int(card.resistance),
            int(card.energyType), int(card.basic), int(card.stage1), int(card.stage2),
            int(card.ex), int(card.megaEx), int(card.tera), int(card.aceSpec),
        ])

    for attack in all_attack():
        aid = int(attack.attackId)
        if not 0 <= aid <= attack_vocab_size:
            continue
        cost = [int(x) for x in attack.energies][:max_attack_cost]
        attack_features[aid] = torch.tensor([
            1, int(attack.damage), len(cost), int(bool(attack.text)), len(attack.name), len(attack.text)
        ])
        if cost:
            attack_cost_energy_types[aid, :len(cost)] = torch.tensor(cost)
        attack_cost_counts[aid] = len(cost)

    return {
        "card_features": card_features,
        "attack_features": attack_features,
        "attack_cost_energy_types": attack_cost_energy_types,
        "attack_cost_counts": attack_cost_counts,
    }
