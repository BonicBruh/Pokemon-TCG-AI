"""Coordinated Mega Starmie policy for the cabt simulator.

The simulator supplies only legal options. This module combines the patterns
reconstructed from Yushin Ito's games with replay-tested attack, survival,
Boss, retreat, and board-continuity projections.
"""

from __future__ import annotations

import math
from itertools import combinations
from collections.abc import Iterable

from cg.api import (
    AreaType,
    Card,
    EnergyType,
    LogType,
    Observation,
    Option,
    OptionType,
    Pokemon,
    SelectContext,
    SelectType,
    all_attack,
    all_card_data,
)

from .card_ids import (
    BOSS_ORDERS,
    BUDDY_BUDDY_POFFIN,
    CINDERACE,
    CRUSHING_HAMMER,
    HARLEQUIN,
    HEROS_CAPE,
    HILDA,
    IGNITION_ENERGY,
    JETTING_BLOW,
    LILLIES_DETERMINATION,
    MEGA_SIGNAL,
    MEGA_STARMIE,
    NEBULA_BEAM,
    NIGHT_STRETCHER,
    POKEGEAR,
    SALVATORE,
    STARYU,
    TURBO_FLARE,
    ULTRA_BALL,
    WALLYS_COMPASSION,
    WATER_ENERGY,
    WATER_GUN,
)


CARD_META = {card.cardId: card for card in all_card_data()}
ATTACK_META = {attack.attackId: attack for attack in all_attack()}


def _player(obs: Observation):
    return obs.current.players[obs.current.yourIndex]


def _opponent(obs: Observation):
    return obs.current.players[1 - obs.current.yourIndex]


def _hand(obs: Observation) -> list[Card]:
    return _player(obs).hand or []


def _active(obs: Observation, ours: bool = True) -> Pokemon | None:
    player = _player(obs) if ours else _opponent(obs)
    return player.active[0] if player.active and player.active[0] is not None else None


def _all_ours(obs: Observation) -> list[Pokemon]:
    player = _player(obs)
    active = [player.active[0]] if player.active and player.active[0] is not None else []
    return active + list(player.bench)


def _card_at_area(obs: Observation, area, index: int | None, player_index: int | None) -> Card | Pokemon | None:
    if index is None or obs.current is None:
        return None
    owner = obs.current.yourIndex if player_index is None else player_index
    player = obs.current.players[owner]

    if area == AreaType.HAND:
        hand = player.hand or []
        return hand[index] if 0 <= index < len(hand) else None
    if area == AreaType.DISCARD:
        return player.discard[index] if 0 <= index < len(player.discard) else None
    if area == AreaType.ACTIVE:
        return player.active[index] if 0 <= index < len(player.active) else None
    if area == AreaType.BENCH:
        return player.bench[index] if 0 <= index < len(player.bench) else None
    if area == AreaType.PRIZE:
        return player.prize[index] if 0 <= index < len(player.prize) else None
    if area == AreaType.STADIUM:
        return obs.current.stadium[index] if 0 <= index < len(obs.current.stadium) else None
    if area == AreaType.LOOKING and obs.current.looking is not None:
        return obs.current.looking[index] if 0 <= index < len(obs.current.looking) else None
    if area == AreaType.DECK and obs.select.deck is not None:
        return obs.select.deck[index] if 0 <= index < len(obs.select.deck) else None
    return None


def _option_card(obs: Observation, option: Option) -> Card | Pokemon | None:
    if option.type in (OptionType.PLAY, OptionType.ATTACH, OptionType.EVOLVE):
        hand = _hand(obs)
        return hand[option.index] if option.index is not None and 0 <= option.index < len(hand) else None
    if option.cardId is not None:
        owner = obs.current.yourIndex if option.playerIndex is None else option.playerIndex
        return Card(option.cardId, option.serial or -1, owner)
    return _card_at_area(obs, option.area, option.index, option.playerIndex)


def _option_card_id(obs: Observation, option: Option) -> int | None:
    card = _option_card(obs, option)
    return card.id if card is not None else option.cardId


def _option_target(obs: Observation, option: Option) -> Pokemon | None:
    if option.inPlayArea is None or option.inPlayIndex is None:
        card = _card_at_area(obs, option.area, option.index, option.playerIndex)
        return card if isinstance(card, Pokemon) else None
    owner = obs.current.yourIndex
    card = _card_at_area(obs, option.inPlayArea, option.inPlayIndex, owner)
    return card if isinstance(card, Pokemon) else None


def _has_card(obs: Observation, card_id: int) -> bool:
    return any(card.id == card_id for card in _hand(obs))


def _in_play(obs: Observation, card_id: int) -> list[Pokemon]:
    return [pokemon for pokemon in _all_ours(obs) if pokemon.id == card_id]


def _has_energy(pokemon: Pokemon, card_id: int | None = None) -> bool:
    if card_id is None:
        return bool(pokemon.energyCards)
    return any(card.id == card_id for card in pokemon.energyCards)


def _has_water(pokemon: Pokemon) -> bool:
    return EnergyType.WATER in pokemon.energies or _has_energy(pokemon, WATER_ENERGY)


def _ready_starmie(obs: Observation) -> list[Pokemon]:
    return [pokemon for pokemon in _in_play(obs, MEGA_STARMIE) if _has_water(pokemon) or len(pokemon.energies) >= 3]


def _damage(pokemon: Pokemon) -> int:
    return max(0, pokemon.maxHp - pokemon.hp)


def _prize_value(card_id: int) -> int:
    meta = CARD_META.get(card_id)
    if meta is None:
        return 1
    if meta.megaEx:
        return 3
    if meta.ex:
        return 2
    return 1


def _water_attack_damage(defender: Pokemon | None, base: int, ignore_modifiers: bool = False) -> int:
    if defender is None or ignore_modifiers:
        return base
    meta = CARD_META.get(defender.id)
    if meta is not None and meta.weakness == EnergyType.WATER:
        return base * 2
    return base


def _best_bench_target(obs: Observation) -> tuple[Pokemon | None, float]:
    """Return the best Jetting Blow target and its tactical value."""
    best: Pokemon | None = None
    best_score = -1.0
    for pokemon in _opponent(obs).bench:
        meta = CARD_META.get(pokemon.id)
        score = 0.0

        if pokemon.hp <= 50:
            score += 500 + 100 * _prize_value(pokemon.id)
        if meta is not None:
            if meta.basic:
                score += 90
            if meta.stage1 or meta.stage2:
                score += 55
            if meta.megaEx:
                score += 100
            elif meta.ex:
                score += 70
        if pokemon.energyCards:
            score += 35 + 8 * len(pokemon.energyCards)
        if pokemon.hp <= 170:
            score += 30
        if pokemon.hp <= 260:
            score += 15

        # Known evolution choke points from Ito's observed targeting.
        if CARD_META.get(pokemon.id) and CARD_META[pokemon.id].name in {
            "Abra",
            "Dreepy",
            "Drakloak",
            "Staryu",
            "Snorunt",
            "Riolu",
            "Makuhita",
            "Hop's Phantump",
            "Dunsparce",
            "Budew",
        }:
            score += 160

        if score > best_score:
            best, best_score = pokemon, score
    return best, best_score


def _can_take_final_prizes(obs: Observation, target: Pokemon, damage: int) -> bool:
    return target.hp <= damage and len(_player(obs).prize) <= _prize_value(target.id)


def _is_engine_target(pokemon: Pokemon) -> bool:
    meta = CARD_META.get(pokemon.id)
    return bool(
        meta
        and meta.name
        in {
            "Drakloak",
            "Abra",
            "Riolu",
            "Staryu",
            "Lunatone",
            "Dunsparce",
            "Dudunsparce",
            "Budew",
        }
    )


def _available_attack_ids(obs: Observation) -> list[int]:
    """Return attacks the current Active can credibly use right now."""
    active = _active(obs)
    if active is None:
        return []

    if obs.select is not None and obs.select.type == SelectType.MAIN:
        legal = [
            option.attackId
            for option in obs.select.option
            if option.type == OptionType.ATTACK and option.attackId is not None
        ]
        if legal:
            return legal

    if active.id == MEGA_STARMIE:
        attacks = []
        if _has_water(active):
            attacks.append(JETTING_BLOW)
        if len(active.energies) >= 3:
            attacks.append(NEBULA_BEAM)
        return attacks
    if active.id == CINDERACE and active.energyCards:
        return [TURBO_FLARE]
    if active.id == STARYU and _has_water(active):
        return [WATER_GUN]
    return []


def _project_attack_outcome(
    obs: Observation,
    active_target: Pokemon,
    bench_targets: list[Pokemon],
) -> tuple[bool, bool, int, int]:
    """Project (win, board-out, Prizes, engine KOs) after the best attack."""
    best = (False, False, 0, 0)
    for attack_id in _available_attack_ids(obs):
        active_ko = False
        bench_ko: Pokemon | None = None

        if attack_id == JETTING_BLOW:
            active_ko = active_target.hp <= _water_attack_damage(active_target, 120)
            legal_kos = [pokemon for pokemon in bench_targets if pokemon.hp <= 50]
            if legal_kos:
                bench_ko = max(
                    legal_kos,
                    key=lambda pokemon: (
                        _prize_value(pokemon.id),
                        int(_is_engine_target(pokemon)),
                        len(pokemon.energyCards),
                    ),
                )
        elif attack_id == NEBULA_BEAM:
            active_ko = active_target.hp <= 210
        else:
            attack = ATTACK_META.get(attack_id)
            active_ko = bool(attack and active_target.hp <= attack.damage)

        prizes = _prize_value(active_target.id) if active_ko else 0
        engines = int(active_ko and _is_engine_target(active_target))
        if bench_ko is not None:
            prizes += _prize_value(bench_ko.id)
            engines += int(_is_engine_target(bench_ko))

        remaining_bench = len(bench_targets) - int(bench_ko is not None)
        board_out = active_ko and remaining_bench == 0
        win = board_out or prizes >= len(_player(obs).prize)
        outcome = (win, board_out, prizes, engines)
        if outcome > best:
            best = outcome
    return best


def _attack_damage(attack_id: int, target: Pokemon) -> int:
    if attack_id == JETTING_BLOW:
        return _water_attack_damage(target, 120)
    if attack_id == NEBULA_BEAM:
        return 210
    attack = ATTACK_META.get(attack_id)
    return attack.damage if attack is not None else 0


def _attack_route_value(
    obs: Observation,
    active_target: Pokemon,
    bench_targets: list[Pokemon],
) -> float:
    """Estimate this turn plus damage progress toward the next Prize turn.

    Immediate-Prize comparison alone made Boss abandon 210 damage on a
    three-Prize Mega for a one-Prize Basic. Fractional Prize progress keeps
    that two-turn route visible without requiring a full game-tree search.
    """
    best = -1.0
    for attack_id in _available_attack_ids(obs):
        damage = _attack_damage(attack_id, active_target)
        active_ko = active_target.hp <= damage
        active_prizes = _prize_value(active_target.id)
        active_progress = (
            float(active_prizes)
            if active_ko
            else active_prizes * min(0.95, damage / max(1, active_target.hp))
        )
        engine_value = 0.25 if active_ko and _is_engine_target(active_target) else 0.0
        bench_progress = 0.0
        bench_ko: Pokemon | None = None

        if attack_id == JETTING_BLOW and bench_targets:
            def bench_value(pokemon: Pokemon) -> float:
                prizes = _prize_value(pokemon.id)
                progress = (
                    float(prizes)
                    if pokemon.hp <= 50
                    else prizes * min(0.8, 50 / max(1, pokemon.hp))
                )
                if pokemon.hp <= 50 and _is_engine_target(pokemon):
                    progress += 0.25
                return progress

            bench_ko = max(bench_targets, key=bench_value)
            bench_progress = bench_value(bench_ko)

        remaining_bench = len(bench_targets) - int(
            bench_ko is not None and bench_ko.hp <= 50
        )
        board_out = active_ko and remaining_bench == 0
        prizes = active_prizes if active_ko else 0
        if bench_ko is not None and bench_ko.hp <= 50:
            prizes += _prize_value(bench_ko.id)
        win = board_out or prizes >= len(_player(obs).prize)

        value = 1000 * (active_progress + bench_progress + engine_value)
        if board_out:
            value += 3000
        if win:
            value += 10000
        best = max(best, value)
    return best


def _boss_candidate_outcome(obs: Observation, target: Pokemon) -> tuple[bool, bool, int, int]:
    old_active = _active(obs, ours=False)
    remaining_bench = [
        pokemon for pokemon in _opponent(obs).bench if pokemon.serial != target.serial
    ]
    if old_active is not None:
        remaining_bench.append(old_active)
    return _project_attack_outcome(obs, target, remaining_bench)


def _boss_candidate_route_value(obs: Observation, target: Pokemon) -> float:
    old_active = _active(obs, ours=False)
    remaining_bench = [
        pokemon for pokemon in _opponent(obs).bench if pokemon.serial != target.serial
    ]
    if old_active is not None:
        remaining_bench.append(old_active)
    return _attack_route_value(obs, target, remaining_bench)


def _pokemon_attack_damage(
    attacker: Pokemon,
    defender: Pokemon | None,
    *,
    next_turn: bool = False,
    blocked_attack: str | None = None,
) -> int:
    """Estimate the largest credible attack from one Pokémon.

    For next-turn threat checks, permit one ordinary attachment. This is
    intentionally conservative: preserving a three-Prize Mega is worth more
    than squeezing a marginal Supporter turn.
    """
    meta = CARD_META.get(attacker.id)
    if meta is None or defender is None:
        return 0

    available = len(attacker.energies) + (1 if next_turn else 0)
    best = 0
    for attack_id in meta.attacks:
        attack = ATTACK_META.get(attack_id)
        if (
            attack is None
            or attack.name == blocked_attack
            or len(attack.energies) > available
        ):
            continue
        damage = attack.damage
        defender_meta = CARD_META.get(defender.id)
        if (
            damage
            and defender_meta is not None
            and defender_meta.weakness == meta.energyType
            and "isn’t affected by Weakness" not in attack.text
        ):
            damage *= 2
        best = max(best, damage)
    return best


def _credible_return_damage(obs: Observation) -> int:
    defender = _active(obs)
    if defender is None:
        return 0

    opponent = _opponent(obs)
    candidates = [
        pokemon
        for pokemon in list(opponent.active) + list(opponent.bench)
        if pokemon is not None
    ]
    locked_lucario_serials = {
        log.serial
        for log in obs.logs
        if log.type == LogType.ATTACK
        and log.playerIndex == 1 - obs.current.yourIndex
        and log.attackId in ATTACK_META
        and ATTACK_META[log.attackId].name == "Mega Brave"
        and log.serial is not None
    }
    best = max(
        (
            _pokemon_attack_damage(
                pokemon,
                defender,
                next_turn=True,
                blocked_attack=(
                    "Mega Brave" if pokemon.serial in locked_lucario_serials else None
                ),
            )
            for pokemon in candidates
        ),
        default=0,
    )

    # Archetype-aware ceilings cover common one-turn acceleration that a
    # simple attached-Energy count cannot see.
    ids = {pokemon.id for pokemon in candidates}
    names = {CARD_META[pokemon_id].name for pokemon_id in ids if pokemon_id in CARD_META}
    unlocked_lucario = any(
        CARD_META.get(pokemon.id)
        and CARD_META[pokemon.id].name == "Mega Lucario ex"
        and pokemon.serial not in locked_lucario_serials
        for pokemon in candidates
    )
    if unlocked_lucario:
        best = max(best, 270)
    if "Dragapult ex" in names:
        best = max(best, 200)
    if "Mega Starmie ex" in names:
        best = max(best, 210)
    return best


def _opponent_attack_prizes(
    obs: Observation,
    attacker: Pokemon,
    defender: Pokemon,
    bench_targets: list[Pokemon],
) -> int:
    """Project the largest one-attack Prize take, including known spread."""
    meta = CARD_META.get(attacker.id)
    if meta is None:
        return 0

    locked_mega_brave = any(
        log.type == LogType.ATTACK
        and log.playerIndex == 1 - obs.current.yourIndex
        and log.serial == attacker.serial
        and log.attackId in ATTACK_META
        and ATTACK_META[log.attackId].name == "Mega Brave"
        for log in obs.logs
    )
    available = len(attacker.energies) + 1
    best = 0

    for attack_id in meta.attacks:
        attack = ATTACK_META.get(attack_id)
        if attack is None or len(attack.energies) > available:
            continue
        if locked_mega_brave and attack.name == "Mega Brave":
            continue

        damage = attack.damage
        defender_meta = CARD_META.get(defender.id)
        if (
            damage
            and defender_meta is not None
            and defender_meta.weakness == meta.energyType
            and "isnâ€™t affected by Weakness" not in attack.text
        ):
            damage *= 2

        prizes = _prize_value(defender.id) if defender.hp <= damage else 0

        if attack.name == "Jetting Blow":
            bench_kos = [
                _prize_value(pokemon.id)
                for pokemon in bench_targets
                if pokemon.hp <= 50
            ]
            prizes += max(bench_kos, default=0)

        if attack.name == "Phantom Dive":
            # Six damage counters may be split in any way. Maximize Prize
            # yield among Pokémon that can be fully removed by those counters.
            eligible = [
                pokemon for pokemon in bench_targets if pokemon.hp <= 60
            ]
            spread_prizes = 0
            for count in range(1, len(eligible) + 1):
                for subset in combinations(eligible, count):
                    counters = sum(math.ceil(pokemon.hp / 10) for pokemon in subset)
                    if counters <= 6:
                        spread_prizes = max(
                            spread_prizes,
                            sum(_prize_value(pokemon.id) for pokemon in subset),
                        )
            prizes += spread_prizes

        best = max(best, prizes)
    return best


def _projected_opponent_prizes(
    obs: Observation,
    defender: Pokemon,
    bench_targets: list[Pokemon] | None = None,
    opponent_attackers: list[Pokemon] | None = None,
) -> int:
    if bench_targets is None:
        bench_targets = [
            pokemon
            for pokemon in _all_ours(obs)
            if pokemon.serial != defender.serial
        ]
    if opponent_attackers is None:
        opponent_attackers = [
            pokemon
            for pokemon in list(_opponent(obs).active) + list(_opponent(obs).bench)
            if pokemon is not None
        ]
    return max(
        (
            _opponent_attack_prizes(
                obs,
                attacker,
                defender,
                bench_targets,
            )
            for attacker in opponent_attackers
        ),
        default=0,
    )


def _attack_stabilizes(obs: Observation, attack_id: int) -> bool:
    """Whether this attack removes the threat without a dangerous replacement."""
    opponent_active = _active(obs, ours=False)
    active = _active(obs)
    if opponent_active is None or active is None:
        return False
    if opponent_active.hp > _attack_damage(attack_id, opponent_active):
        return False

    remaining = list(_opponent(obs).bench)
    if attack_id == JETTING_BLOW:
        bench_target, _ = _best_bench_target(obs)
        if bench_target is not None and bench_target.hp <= 50:
            remaining = [
                pokemon
                for pokemon in remaining
                if pokemon.serial != bench_target.serial
            ]
    if not remaining:
        return True

    projected = _projected_opponent_prizes(
        obs,
        active,
        opponent_attackers=remaining,
    )
    return projected == 0


def _active_is_threatened(obs: Observation) -> bool:
    active = _active(obs)
    return bool(
        active
        and active.id == MEGA_STARMIE
        and active.hp <= _credible_return_damage(obs)
    )


def _pokemon_wins_if_promoted(obs: Observation, pokemon: Pokemon) -> bool:
    opponent_active = _active(obs, ours=False)
    if opponent_active is None:
        return False
    attacks: list[int] = []
    if pokemon.id == MEGA_STARMIE:
        if _has_water(pokemon):
            attacks.append(JETTING_BLOW)
        if len(pokemon.energies) >= 3:
            attacks.append(NEBULA_BEAM)
    elif pokemon.id == CINDERACE and pokemon.energyCards:
        attacks.append(TURBO_FLARE)
    elif pokemon.id == STARYU and _has_water(pokemon):
        attacks.append(WATER_GUN)

    for attack_id in attacks:
        active_ko = opponent_active.hp <= _attack_damage(attack_id, opponent_active)
        prizes = _prize_value(opponent_active.id) if active_ko else 0
        bench_ko = False
        if attack_id == JETTING_BLOW:
            target, _ = _best_bench_target(obs)
            bench_ko = target is not None and target.hp <= 50
            if bench_ko:
                prizes += _prize_value(target.id)
        board_out = active_ko and len(_opponent(obs).bench) - int(bench_ko) == 0
        if board_out or prizes >= len(_player(obs).prize):
            return True
    return False


def _cinderace_sacrifice_lock(obs: Observation) -> bool:
    """Keep an intentional one-Prize shield Active until it is removed."""
    active = _active(obs)
    if (
        active is None
        or active.id != CINDERACE
        or not _player(obs).bench
        or len(_opponent(obs).prize) > 3
    ):
        return False
    if _pokemon_wins_if_promoted(obs, active):
        return False

    current_yield = _projected_opponent_prizes(obs, active)
    if current_yield == 0 or current_yield >= len(_opponent(obs).prize):
        return False

    replacement_yields = []
    for replacement in _player(obs).bench:
        if _pokemon_wins_if_promoted(obs, replacement):
            return False
        after_bench = [
            pokemon
            for pokemon in [active] + list(_player(obs).bench)
            if pokemon.serial != replacement.serial
        ]
        replacement_yields.append(
            _projected_opponent_prizes(
                obs,
                replacement,
                bench_targets=after_bench,
            )
        )
    return bool(replacement_yields and current_yield < min(replacement_yields))


def _spread_threat(obs: Observation) -> bool:
    opponent = _opponent(obs)
    names = {
        CARD_META[pokemon.id].name
        for pokemon in list(opponent.active) + list(opponent.bench)
        if pokemon is not None and pokemon.id in CARD_META
    }
    return bool(names & {"Dragapult ex", "Mega Starmie ex", "Mega Froslass ex"})


def _line_count(obs: Observation) -> int:
    return len(_in_play(obs, STARYU)) + len(_in_play(obs, MEGA_STARMIE))


def _needs_reserve(obs: Observation) -> bool:
    if len(_all_ours(obs)) <= 1:
        return True
    desired = 3 if _spread_threat(obs) else 2
    return _line_count(obs) < desired


def _nebula_reduces_clock(obs: Observation) -> bool:
    target = _active(obs, ours=False)
    if target is None:
        return False
    jet = max(1, _water_attack_damage(target, 120))
    return math.ceil(target.hp / 210) < math.ceil(target.hp / jet)


def _should_wally(obs: Observation) -> bool:
    active = _active(obs)
    if active is None or active.id != MEGA_STARMIE or _damage(active) <= 0:
        return False
    if active.hp <= _credible_return_damage(obs):
        return True
    return _damage(active) >= 200 or active.hp <= 120


def _supporter_score(obs: Observation, card_id: int) -> float:
    player = _player(obs)
    active = _active(obs)
    opponent = _opponent(obs)

    if card_id == WALLYS_COMPASSION:
        if not _should_wally(obs):
            return -500
        return 1300 + (_damage(active) if active else 0)

    if card_id == BOSS_ORDERS:
        if not opponent.bench:
            return -500
        opponent_active = _active(obs, ours=False)
        if active is None or opponent_active is None:
            return -500

        direct = _project_attack_outcome(obs, opponent_active, list(opponent.bench))
        candidates = [_boss_candidate_outcome(obs, target) for target in opponent.bench]
        best = max(candidates, default=(False, False, 0, 0))
        direct_route = _attack_route_value(
            obs,
            opponent_active,
            list(opponent.bench),
        )
        best_route = max(
            (_boss_candidate_route_value(obs, target) for target in opponent.bench),
            default=-1.0,
        )

        if best[0] and not direct[0]:
            return 1400
        # A small current-turn edge is not enough: gusting resets damage on
        # the valuable Active and spends the Supporter. Require roughly
        # three-quarters of a Prize of two-turn improvement.
        if best_route >= direct_route + 750:
            return 980 + min(300, (best_route - direct_route) / 5)
        if (
            best[2] == direct[2]
            and best[3] > direct[3]
            and best_route >= direct_route - 100
        ):
            return 860
        # A gust must improve the two-turn route, not merely turn a valuable
        # Active damage target into a one-Prize knockout.
        return -300

    if card_id == SALVATORE:
        staryu = _in_play(obs, STARYU)
        if not staryu:
            return -400
        second_player_first_turn = (
            obs.current.turn == 2 and obs.current.firstPlayer != obs.current.yourIndex
        )
        new_staryu = any(pokemon.appearThisTurn for pokemon in staryu)
        powered_target = any(_has_water(pokemon) or _has_card(obs, WATER_ENERGY) for pokemon in staryu)
        immediate = powered_target and (second_player_first_turn or new_staryu)
        return 1200 if immediate else 620 if new_staryu else 260

    if card_id == HILDA:
        has_starmie_line = bool(_in_play(obs, STARYU) or _in_play(obs, MEGA_STARMIE))
        if not has_starmie_line:
            return -400
        missing_starmie = bool(_in_play(obs, STARYU)) and not _has_card(obs, MEGA_STARMIE)
        missing_energy = not _has_card(obs, WATER_ENERGY) and not _has_card(obs, IGNITION_ENERGY)
        reserve_value = _needs_reserve(obs) and bool(_in_play(obs, STARYU))
        if missing_starmie and missing_energy:
            return 1100
        if missing_starmie or reserve_value:
            return 900
        if _nebula_reduces_clock(obs) and not _has_card(obs, IGNITION_ENERGY):
            return 820
        return 280

    if card_id == LILLIES_DETERMINATION:
        # Eight fresh cards at six prizes is the premier setup Supporter.
        if len(player.prize) == 6:
            return 850 if player.handCount <= 7 else 580
        return 680 if player.handCount <= 4 else 300

    if card_id == HARLEQUIN:
        hand_delta = opponent.handCount - player.handCount
        return 460 + 35 * hand_delta if opponent.handCount >= 5 else 180

    return 0


def _play_score(obs: Observation, card_id: int) -> float:
    player = _player(obs)
    active = _active(obs)
    hand_ids = [card.id for card in _hand(obs)]

    if card_id in {
        WALLYS_COMPASSION,
        BOSS_ORDERS,
        SALVATORE,
        HILDA,
        LILLIES_DETERMINATION,
        HARLEQUIN,
    }:
        return _supporter_score(obs, card_id)

    if card_id == STARYU:
        reserve_needed = _needs_reserve(obs)
        return 1200 if reserve_needed else -100

    if card_id == BUDDY_BUDDY_POFFIN:
        space = player.benchMax - len(player.bench)
        if space <= 0:
            return -500
        if len(_all_ours(obs)) <= 1:
            return 1300
        return 980 if _needs_reserve(obs) else -150

    if card_id == MEGA_SIGNAL:
        if not _in_play(obs, STARYU) or MEGA_STARMIE in hand_ids:
            return -100
        return 1030

    if card_id == POKEGEAR:
        return 820 if not obs.current.supporterPlayed else 260

    if card_id == CRUSHING_HAMMER:
        opposing_energy = sum(len(p.energyCards) for p in (_opponent(obs).active + _opponent(obs).bench) if p)
        return 900 if opposing_energy else 260

    if card_id == NIGHT_STRETCHER:
        discard_ids = {card.id for card in player.discard}
        if STARYU in discard_ids and _needs_reserve(obs):
            return 950
        if WATER_ENERGY in discard_ids and not _has_card(obs, WATER_ENERGY):
            return 760
        if MEGA_STARMIE in discard_ids and _in_play(obs, STARYU):
            return 720
        return -200

    if card_id == ULTRA_BALL:
        if player.handCount < 4:
            return -200
        if len(_all_ours(obs)) <= 1:
            return 1300
        if _in_play(obs, STARYU) and MEGA_STARMIE not in hand_ids:
            return 700
        if not _in_play(obs, STARYU) and STARYU not in hand_ids:
            return 660
        return 100

    if card_id == CINDERACE:
        return -500

    # Tools and Energy normally appear as ATTACH options rather than PLAY.
    return 0


def _attach_score(obs: Observation, option: Option) -> float:
    card_id = _option_card_id(obs, option)
    target = _option_target(obs, option)
    if target is None:
        return -1000
    active = _active(obs)
    is_active = active is not None and target.serial == active.serial
    first_player_first_turn = (
        obs.current.turn == 1 and obs.current.firstPlayer == obs.current.yourIndex
    )
    can_attack_this_turn = not first_player_first_turn

    if card_id == HEROS_CAPE:
        if target.id == MEGA_STARMIE:
            return 1250 + (100 if is_active else 0)
        if target.id == STARYU:
            return 1120 + (60 if _has_water(target) else 0)
        # Never consume the ACE SPEC on Cinderace.
        return -10000

    if card_id == WATER_ENERGY:
        if target.id == MEGA_STARMIE:
            if _has_water(target):
                if (
                    is_active
                    and len(target.energies) < 3
                    and _nebula_reduces_clock(obs)
                ):
                    return 820
                return -150
            if not is_active and active is not None and active.id == CINDERACE:
                # Cinderace retreats for free. Power the finished attacker
                # directly instead of spending the turn on Turbo Flare.
                return 1500
            return 1150 if is_active else 990
        if target.id == STARYU:
            if _has_water(target):
                return -200
            return 1010 if active and active.id == CINDERACE else 900
        if target.id == CINDERACE:
            if not is_active:
                return -10000
            if target.energyCards:
                return -500
            # Power Turbo Flare when Starmie is not already ready.
            if (
                can_attack_this_turn
                and not _ready_starmie(obs)
                and any(p.id == STARYU for p in _player(obs).bench)
            ):
                return 1180
            return -250

    if card_id == IGNITION_ENERGY:
        if target.id == MEGA_STARMIE:
            if not is_active:
                if active is not None and active.id == CINDERACE:
                    opponent_active = _active(obs, ours=False)
                    if opponent_active is not None:
                        jet = _water_attack_damage(opponent_active, 120)
                        if opponent_active.hp <= 210 and opponent_active.hp > jet:
                            return 1600
                        if math.ceil(opponent_active.hp / 210) < math.ceil(
                            opponent_active.hp / max(1, jet)
                        ):
                            return 1520
                    return 1380
                return 160
            opponent_active = _active(obs, ours=False)
            bench_target, _ = _best_bench_target(obs)
            needs_beam = _nebula_reduces_clock(obs)
            jet_damage = _water_attack_damage(opponent_active, 120)
            beam_knockout = bool(
                opponent_active
                and opponent_active.hp <= 210
                and opponent_active.hp > jet_damage
            )
            no_useful_bench = bench_target is None
            if beam_knockout:
                return 1450
            return 1280 if needs_beam else 780 if no_useful_bench else 470
        if target.id == CINDERACE and not target.energyCards and not _ready_starmie(obs):
            if not is_active:
                return -10000
            return 1060 if can_attack_this_turn and any(
                p.id in (STARYU, MEGA_STARMIE) and not _has_water(p)
                for p in _player(obs).bench
            ) else -400
        return -400

    return -100


def _evolve_score(obs: Observation, option: Option) -> float:
    card_id = _option_card_id(obs, option)
    target = _option_target(obs, option)
    if card_id != MEGA_STARMIE or target is None or target.id != STARYU:
        return 0
    active = _active(obs)
    score = 1000
    if active and target.serial == active.serial:
        score += 180
    if _has_water(target):
        score += 160
    if target.tools:
        score += 80
    if _spread_threat(obs):
        score += 180
    return score


def _retreat_score(obs: Observation) -> float:
    active = _active(obs)
    if active is None:
        return -1000
    ready = _ready_starmie(obs)
    if active.id == CINDERACE:
        if _cinderace_sacrifice_lock(obs):
            return -1200
        if ready:
            return 1450
    if active.id == STARYU and ready:
        return 920
    if active.id == MEGA_STARMIE:
        if any(
            option.type == OptionType.ATTACK
            and option.attackId is not None
            and _attack_stabilizes(obs, option.attackId)
            for option in (obs.select.option if obs.select is not None else [])
        ):
            return -500

        threat = _credible_return_damage(obs)
        backups = list(_player(obs).bench)
        projected = _projected_opponent_prizes(obs, active)
        game_losing_attack = projected >= len(_opponent(obs).prize)
        board_out = active.hp <= threat and not backups

        if backups and (game_losing_attack or board_out):
            current_bench = list(backups)
            best_replacement_yield = min(
                _projected_opponent_prizes(
                    obs,
                    pokemon,
                    bench_targets=[
                        candidate
                        for candidate in [active] + current_bench
                        if candidate.serial != pokemon.serial
                    ],
                )
                for pokemon in backups
            )
            if best_replacement_yield < projected:
                return 1700

        if backups and active.hp <= threat and len(_opponent(obs).prize) <= 3:
            return 1500

        fresher = [p for p in ready if p.serial != active.serial and p.hp > active.hp]
        return 820 if fresher and active.hp <= 120 else -300
    return -200


def _attack_score(obs: Observation, attack_id: int) -> float:
    opponent_active = _active(obs, ours=False)
    if attack_id == JETTING_BLOW:
        damage = _water_attack_damage(opponent_active, 120)
        score = 700
        if opponent_active is not None and opponent_active.hp <= damage:
            score += 550 + 120 * _prize_value(opponent_active.id)
        bench, bench_score = _best_bench_target(obs)
        if bench is not None:
            score += min(420, bench_score)
            if bench.hp <= 50:
                score += 360 + 100 * _prize_value(bench.id)
        if opponent_active is not None:
            jet_hits = math.ceil(opponent_active.hp / max(1, damage))
            beam_hits = math.ceil(opponent_active.hp / 210)
            if beam_hits < jet_hits:
                score -= 320 * (jet_hits - beam_hits)
        return score

    if attack_id == NEBULA_BEAM:
        score = 730
        if opponent_active is not None:
            if opponent_active.hp <= 210:
                score += 560 + 120 * _prize_value(opponent_active.id)
            elif opponent_active.hp > 210:
                score += 120
            jet_damage = _water_attack_damage(opponent_active, 120)
            if opponent_active.hp <= jet_damage:
                score -= 420  # Jetting Blow already gets the Active knockout.
            else:
                jet_hits = math.ceil(opponent_active.hp / max(1, jet_damage))
                beam_hits = math.ceil(opponent_active.hp / 210)
                if beam_hits < jet_hits:
                    score += 520 * (jet_hits - beam_hits)
        if not _opponent(obs).bench:
            score += 190
        return score

    if attack_id == TURBO_FLARE:
        productive_target = any(
            pokemon.id in (STARYU, MEGA_STARMIE) and not _has_water(pokemon)
            for pokemon in _player(obs).bench
        )
        return 760 if productive_target else 260

    if attack_id == WATER_GUN:
        return 350
    return 100


def _attack_is_immediate_win(obs: Observation, attack_id: int) -> bool:
    active = _active(obs, ours=False)
    if active is None:
        return False
    bench = None
    bench_ko = False
    if attack_id == JETTING_BLOW:
        active_ko = active.hp <= _water_attack_damage(active, 120)
        bench, _ = _best_bench_target(obs)
        bench_ko = bench is not None and bench.hp <= 50
    elif attack_id == NEBULA_BEAM:
        active_ko = active.hp <= 210
    else:
        meta = ATTACK_META.get(attack_id)
        active_ko = bool(meta and active.hp <= meta.damage)

    prizes = (_prize_value(active.id) if active_ko else 0)
    if bench_ko and bench is not None:
        prizes += _prize_value(bench.id)
    board_out = active_ko and len(_opponent(obs).bench) - int(bench_ko) <= 0
    return board_out or prizes >= len(_player(obs).prize)


def _main_priority(obs: Observation, option: Option) -> tuple[int, float]:
    """Return lexicographic phase and within-phase value.

    Attacks are terminal. They intentionally sit below survival, board
    continuity, attacker completion, and useful setup unless they win now.
    """
    if option.type == OptionType.ATTACK:
        attack_id = option.attackId or -1
        if _attack_is_immediate_win(obs, attack_id):
            return 1000, _attack_score(obs, attack_id)
        if _active_is_threatened(obs) and _attack_stabilizes(obs, attack_id):
            # Removing the threat is itself a survival action. This prevents
            # "attach Ignition for the knockout, then retreat" contradictions.
            return 850, _attack_score(obs, attack_id)
        return 500, _attack_score(obs, attack_id)

    if option.type == OptionType.PLAY:
        card_id = _option_card_id(obs, option) or -1
        score = _play_score(obs, card_id)

        if card_id == WALLYS_COMPASSION and _should_wally(obs):
            return 950, score
        if card_id == WALLYS_COMPASSION:
            return -100, score

        if card_id in (STARYU, BUDDY_BUDDY_POFFIN, ULTRA_BALL, NIGHT_STRETCHER):
            discard_ids = {card.id for card in _player(obs).discard}
            can_add_reserve = card_id in (
                STARYU,
                BUDDY_BUDDY_POFFIN,
                ULTRA_BALL,
            ) or (card_id == NIGHT_STRETCHER and STARYU in discard_ids)
            if len(_all_ours(obs)) <= 1 and can_add_reserve and score > 0:
                return 900, score

        if card_id == BOSS_ORDERS:
            boss_score = _supporter_score(obs, card_id)
            if boss_score >= 1400:
                return 1000, boss_score
            if boss_score >= 860:
                return 740, boss_score
            if boss_score <= 0:
                return -100, boss_score
            return 430, boss_score

        if card_id == SALVATORE:
            salvatore_score = _supporter_score(obs, card_id)
            if salvatore_score >= 1000:
                return 830, salvatore_score
            if salvatore_score >= 600:
                return 620, salvatore_score
            if salvatore_score <= 0:
                return -100, salvatore_score
            return 420, salvatore_score

        if card_id == HILDA:
            hilda_score = _supporter_score(obs, card_id)
            if hilda_score >= 1000:
                return 780, hilda_score
            if hilda_score >= 820:
                return 700, hilda_score
            if hilda_score <= 0:
                return -100, hilda_score
            return 440, hilda_score

        if card_id == MEGA_SIGNAL:
            lillie_before_search = (
                _has_card(obs, LILLIES_DETERMINATION)
                and not obs.current.supporterPlayed
                and _supporter_score(obs, LILLIES_DETERMINATION) >= 580
                and not any(
                    not pokemon.appearThisTurn for pokemon in _in_play(obs, STARYU)
                )
            )
            if lillie_before_search:
                return 600, score

        if card_id in (STARYU, BUDDY_BUDDY_POFFIN, MEGA_SIGNAL, NIGHT_STRETCHER, ULTRA_BALL):
            return (720 if score > 0 else -100), score

        if card_id in (POKEGEAR, CRUSHING_HAMMER):
            return (650 if score > 0 else -100), score

        if card_id == LILLIES_DETERMINATION:
            if len(_all_ours(obs)) <= 1 and score >= 580:
                return 880, score
            return (680 if score >= 600 else 440), score

        if card_id == HARLEQUIN:
            return (640 if score >= 450 else 420), score

        return (400 if score > 0 else -100), score

    if option.type == OptionType.EVOLVE:
        return 820, _evolve_score(obs, option)

    if option.type == OptionType.ATTACH:
        attach_score = _attach_score(obs, option)
        return (810 if attach_score > 0 else -100), attach_score

    if option.type == OptionType.RETREAT:
        retreat_score = _retreat_score(obs)
        return (800 if retreat_score > 0 else -100), retreat_score

    if option.type == OptionType.ABILITY:
        return 660, 700

    if option.type == OptionType.END:
        return 0, 0
    return 100, 0


def _discard_value(card_id: int) -> float:
    """Lower values are safer to discard."""
    return {
        CINDERACE: 15,
        HARLEQUIN: 20,
        CRUSHING_HAMMER: 30,
        POKEGEAR: 35,
        BUDDY_BUDDY_POFFIN: 40,
        SALVATORE: 45,
        MEGA_SIGNAL: 50,
        LILLIES_DETERMINATION: 55,
        HILDA: 60,
        BOSS_ORDERS: 65,
        IGNITION_ENERGY: 70,
        NIGHT_STRETCHER: 75,
        WATER_ENERGY: 80,
        STARYU: 90,
        MEGA_STARMIE: 95,
        WALLYS_COMPASSION: 100,
        HEROS_CAPE: 110,
    }.get(card_id, 25)


def _boss_target_score(obs: Observation, target: Pokemon) -> float:
    """Rank an opponent's Bench Pokémon after Boss's Orders is played."""
    win, board_out, prizes, engine_kos = _boss_candidate_outcome(obs, target)
    route_value = _boss_candidate_route_value(obs, target)
    score = (
        10000 * int(win)
        + 3000 * int(board_out)
        + 1000 * prizes
        + 250 * engine_kos
        + route_value
    )

    meta = CARD_META.get(target.id)
    if meta is not None:
        score += 25 * meta.retreatCost
        if meta.megaEx:
            score += 170
        elif meta.ex:
            score += 100

    score += 25 * len(target.energyCards)
    score += max(0, 220 - target.hp) / 10
    return score


def _card_choice_score(obs: Observation, option: Option) -> float:
    context = obs.select.context
    effect_id = obs.select.effect.id if obs.select.effect is not None else None
    card_id = _option_card_id(obs, option)
    card = _option_card(obs, option)
    pokemon = card if isinstance(card, Pokemon) else None

    if context == SelectContext.SETUP_ACTIVE_POKEMON:
        second_player = obs.current.firstPlayer != obs.current.yourIndex
        fast_starmie = (
            second_player
            and _has_card(obs, SALVATORE)
            and (_has_card(obs, WATER_ENERGY) or _has_card(obs, IGNITION_ENERGY))
        )
        if card_id == STARYU:
            return 1180 if fast_starmie else 860
        if card_id == CINDERACE:
            return 1000
        return 0

    if context == SelectContext.SETUP_BENCH_POKEMON:
        return 900 if card_id == STARYU else -100

    if context in (SelectContext.SWITCH, SelectContext.TO_ACTIVE):
        if pokemon is None:
            return 0
        if option.playerIndex is not None and option.playerIndex != obs.current.yourIndex:
            return _boss_target_score(obs, pokemon)
        current_active = _active(obs)
        threat = _credible_return_damage(obs)
        current_projected = (
            _projected_opponent_prizes(obs, current_active)
            if current_active is not None
            else 0
        )
        defensive_pivot = bool(
            current_active
            and current_active.id == MEGA_STARMIE
            and current_active.hp <= threat
            and (
                len(_opponent(obs).prize) <= 3
                or current_projected >= len(_opponent(obs).prize)
            )
        )
        after_bench = [
            candidate
            for candidate in [current_active] + list(_player(obs).bench)
            if candidate is not None and candidate.serial != pokemon.serial
        ]
        candidate_projected = (
            _projected_opponent_prizes(
                obs,
                pokemon,
                bench_targets=after_bench,
            )
            if defensive_pivot
            else current_projected
        )
        survival_bonus = (
            1800
            if defensive_pivot
            and current_projected >= len(_opponent(obs).prize)
            and candidate_projected < len(_opponent(obs).prize)
            else 700 * max(0, current_projected - candidate_projected)
        )
        denial_bonus = (
            450
            if defensive_pivot
            and _prize_value(pokemon.id) == 1
            else 0
        )
        if pokemon.id == MEGA_STARMIE:
            return (
                900
                + (500 if _has_water(pokemon) else 0)
                + survival_bonus
                + pokemon.hp / 10
            )
        if pokemon.id == CINDERACE:
            productive_turbo = any(
                target.id in (STARYU, MEGA_STARMIE) and not _has_water(target)
                for target in _player(obs).bench
                if target.serial != pokemon.serial
            )
            return (
                650
                + (260 if pokemon.energyCards and productive_turbo else 0)
                + survival_bonus
                + denial_bonus
                + pokemon.hp / 10
            )
        if pokemon.id == STARYU:
            return (
                420
                + (420 if _has_water(pokemon) else 0)
                + (120 if pokemon.tools else 0)
                + (100 if _has_card(obs, MEGA_STARMIE) else 0)
                + survival_bonus
                + denial_bonus
                + pokemon.hp / 10
            )

    if context in (SelectContext.TO_BENCH, SelectContext.TO_FIELD):
        return 1000 if card_id == STARYU else 500 if card_id == CINDERACE else 0

    if context == SelectContext.DISCARD:
        return -_discard_value(card_id or -1)

    if context == SelectContext.TO_HAND:
        if effect_id == MEGA_SIGNAL:
            return 1200 if card_id == MEGA_STARMIE else 0
        if effect_id == POKEGEAR:
            return _supporter_score(obs, card_id or -1)
        if effect_id == HILDA:
            if card_id == MEGA_STARMIE:
                return 1100
            if card_id == IGNITION_ENERGY:
                return 1080 if _nebula_reduces_clock(obs) else 700
            if card_id == WATER_ENERGY:
                return 900 if not _nebula_reduces_clock(obs) else 820
            if card_id == CINDERACE:
                return 300
        if effect_id == NIGHT_STRETCHER:
            if card_id == STARYU and not _in_play(obs, STARYU):
                return 1100
            if card_id == WATER_ENERGY and not _has_card(obs, WATER_ENERGY):
                return 1000
            if card_id == MEGA_STARMIE and _in_play(obs, STARYU):
                return 900
            return -500 if card_id == CINDERACE else 100
        if effect_id == ULTRA_BALL:
            if len(_all_ours(obs)) <= 1 and card_id == STARYU:
                return 1500
            if _in_play(obs, STARYU) and card_id == MEGA_STARMIE:
                return 1100
            if not _in_play(obs, STARYU) and card_id == STARYU:
                return 1080
            return 500 if card_id in (MEGA_STARMIE, STARYU) else 100
        if card_id == MEGA_STARMIE:
            return 1000
        if card_id == STARYU:
            return 900
        if card_id == WATER_ENERGY:
            return 800

    if context == SelectContext.ATTACH_TO:
        # Turbo Flare chooses Basic Energy from the deck here.
        return 1200 if card_id == WATER_ENERGY else -100

    if context == SelectContext.ATTACH_FROM:
        if pokemon is None:
            return 0
        if pokemon.id == MEGA_STARMIE:
            return 1400 if not _has_water(pokemon) else -200
        if pokemon.id == STARYU:
            return 1200 if not _has_water(pokemon) else -200
        return -500

    if context in (SelectContext.HEAL, SelectContext.REMOVE_DAMAGE_COUNTER):
        if pokemon is None:
            return 0
        active = _active(obs)
        is_active = active is not None and pokemon.serial == active.serial
        threatened = is_active and pokemon.hp <= _credible_return_damage(obs)
        return (
            _damage(pokemon)
            + (700 if pokemon.id == MEGA_STARMIE else 0)
            + (900 if threatened else 250 if is_active else 0)
        )

    if context in (SelectContext.DAMAGE, SelectContext.DAMAGE_COUNTER, SelectContext.EFFECT_TARGET):
        if pokemon is None:
            return 0
        if option.playerIndex == obs.current.yourIndex:
            return -500
        meta = CARD_META.get(pokemon.id)
        score = 500 if pokemon.hp <= 50 else 100
        if meta and meta.basic:
            score += 120
        if pokemon.energyCards:
            score += 60
        return score

    if context in (
        SelectContext.DISCARD_ENERGY_CARD,
        SelectContext.DISCARD_ENERGY,
        SelectContext.DISCARD_CARD_OR_ATTACHED_CARD,
    ):
        # When discarding the opponent's Energy, prefer Special Energy and a
        # developed attacker. When paying our own cost, preserve Ignition only
        # if it is still relevant this turn.
        return 900 if card_id == IGNITION_ENERGY else 700 if card_id == WATER_ENERGY else 0

    if context in (SelectContext.EVOLVES_FROM, SelectContext.EVOLVE):
        return 1000 if card_id == STARYU else 0
    if context == SelectContext.EVOLVES_TO:
        return 1000 if card_id == MEGA_STARMIE else 0

    if context == SelectContext.TO_PRIZE:
        return option.index or 0

    return 0


def _yes_no_choice(obs: Observation) -> list[int]:
    context = obs.select.context
    yes = [i for i, option in enumerate(obs.select.option) if option.type == OptionType.YES]
    no = [i for i, option in enumerate(obs.select.option) if option.type == OptionType.NO]

    choose_yes = True
    if context == SelectContext.IS_FIRST:
        choose_yes = True
    elif context == SelectContext.MULLIGAN:
        ids = {card.id for card in _hand(obs)}
        # Keep either the conventional Basic or an Explosiveness Cinderace.
        choose_yes = STARYU not in ids and CINDERACE not in ids
    elif context == SelectContext.ACTIVATE:
        choose_yes = True

    candidates = yes if choose_yes else no
    if candidates:
        return [candidates[0]]
    return [0] if obs.select.minCount else []


def _choose_indices(
    scores: Iterable[tuple[int, float]],
    minimum: int,
    maximum: int,
    *,
    optional_threshold: float = 0,
    force_exact: int | None = None,
) -> list[int]:
    ranked = sorted(scores, key=lambda pair: (-pair[1], pair[0]))
    desired = force_exact if force_exact is not None else maximum
    desired = max(minimum, min(maximum, desired))

    chosen = [index for index, score in ranked if score > optional_threshold][:desired]
    if len(chosen) < minimum:
        already = set(chosen)
        chosen.extend(index for index, _ in ranked if index not in already and len(chosen) < minimum)
    return chosen


def choose_action(obs: Observation) -> list[int]:
    """Choose legal option indices for every cabt selection context."""
    select = obs.select
    if select is None:
        raise ValueError("Deck selection must be handled by main.agent")
    if not select.option:
        return []

    if select.type == SelectType.YES_NO:
        return _yes_no_choice(obs)

    if select.type == SelectType.MAIN:
        ranked = sorted(
            ((i, _main_priority(obs, option)) for i, option in enumerate(select.option)),
            key=lambda pair: (-pair[1][0], -pair[1][1], pair[0]),
        )
        return [ranked[0][0]]

    if select.type == SelectType.ATTACK:
        scores = [(i, _attack_score(obs, option.attackId or -1)) for i, option in enumerate(select.option)]
        return _choose_indices(scores, select.minCount, select.maxCount, force_exact=1)

    if select.type == SelectType.COUNT:
        scores = [(i, float(option.number or 0)) for i, option in enumerate(select.option)]
        return _choose_indices(scores, select.minCount, select.maxCount, force_exact=1)

    if select.type == SelectType.SPECIAL_CONDITION:
        return list(range(select.minCount))

    scores = [(i, _card_choice_score(obs, option)) for i, option in enumerate(select.option)]

    # Turbo Flare is elastic: take one Water for each distinct, unpowered
    # Staryu/Starmie target. Taking exactly one was an imitation artifact that
    # left recoverable boards needlessly underdeveloped.
    if select.context == SelectContext.ATTACH_TO and select.effect is not None and select.effect.id == CINDERACE:
        useful_targets = sum(
            pokemon.id in (STARYU, MEGA_STARMIE) and not _has_water(pokemon)
            for pokemon in _player(obs).bench
        )
        available_water = sum(
            _option_card_id(obs, option) == WATER_ENERGY for option in select.option
        )
        desired = min(select.maxCount, useful_targets, available_water)
        return _choose_indices(
            scores,
            select.minCount,
            select.maxCount,
            force_exact=desired,
        )

    # Discard costs must satisfy the exact minimum; optional search/bench
    # effects select only strategically useful positive-scoring cards.
    if select.context in {
        SelectContext.DISCARD,
        SelectContext.DISCARD_ENERGY_CARD,
        SelectContext.DISCARD_ENERGY,
        SelectContext.DISCARD_CARD_OR_ATTACHED_CARD,
    }:
        return _choose_indices(scores, select.minCount, select.maxCount, force_exact=select.minCount)

    return _choose_indices(scores, select.minCount, select.maxCount)
