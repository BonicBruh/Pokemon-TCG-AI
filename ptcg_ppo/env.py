from __future__ import annotations
from dataclasses import dataclass
import random
from cg.game import battle_finish, battle_select, battle_start
from opponents import load_opponent

@dataclass
class MatchInfo:
    opponent: str
    learner_slot: int
    winner: int | None
    learner_decisions: int
    total_engine_decisions: int


def _result(obs: dict) -> int:
    current = obs.get("current")
    return -1 if current is None else int(current.get("result", -1))


def _safe_rule_action(agent, obs: dict) -> list[int]:
    select = obs.get("select")
    if select is None:
        return []
    try:
        action = list(map(int, agent(obs)))
    except Exception:
        action = list(range(int(select["minCount"])))
    minimum, maximum = int(select["minCount"]), int(select["maxCount"])
    n_options = len(select["option"])
    valid = minimum <= len(action) <= maximum and len(set(action)) == len(action) and all(0 <= x < n_options for x in action)
    if not valid:
        action = list(range(minimum))
    return action


class CabtMatchEnv:
    """One learner-versus-rule-agent environment backed by the real CABT engine."""
    def __init__(self, learner_deck: list[int], opponent_names=("mega_lucario", "mega_starmie"), *, seed=0,
                 learner_slot: int | None = None, prize_reward=0.12, decision_penalty=0.0005,
                 max_learner_decisions=1500):
        self.learner_deck = list(learner_deck)
        self.opponent_names = tuple(opponent_names)
        self.rng = random.Random(seed)
        self.fixed_learner_slot = learner_slot
        self.prize_reward = float(prize_reward)
        self.decision_penalty = float(decision_penalty)
        self.max_learner_decisions = int(max_learner_decisions)
        self.obs = None
        self.opponent = None
        self.learner_slot = 0
        self.learner_decisions = 0
        self.engine_decisions = 0
        self._prizes = None
        self._started = False

    def close(self):
        if self._started:
            try: battle_finish()
            finally: self._started = False

    def reset(self, *, opponent_name: str | None = None, learner_slot: int | None = None):
        self.close()
        name = opponent_name or self.rng.choice(self.opponent_names)
        self.opponent = load_opponent(name)
        self.opponent.reset()
        self.learner_slot = self.fixed_learner_slot if learner_slot is None else learner_slot
        if self.fixed_learner_slot is None and learner_slot is None:
            self.learner_slot = self.rng.randrange(2)
        if self.learner_slot not in (0, 1):
            raise ValueError("learner_slot must be 0 or 1")
        decks = [self.opponent.deck, self.opponent.deck]
        decks[self.learner_slot] = self.learner_deck
        obs, start = battle_start(decks[0], decks[1])
        if obs is None:
            raise RuntimeError(f"CABT rejected a deck: errorPlayer={start.errorPlayer}, errorType={start.errorType}")
        self._started = True
        self.obs = obs
        self.learner_decisions = 0
        self.engine_decisions = 0
        self._prizes = self._prize_counts(obs)
        terminal_reward = self._advance_opponent()
        if _result(self.obs) >= 0:
            # Extremely rare setup loss; restart to ensure reset returns a live state.
            return self.reset(opponent_name=name, learner_slot=self.learner_slot)
        return self.obs, {"opponent": name, "learner_slot": self.learner_slot, "setup_reward": terminal_reward}

    def _prize_counts(self, obs):
        current = obs.get("current")
        if current is None: return (6, 6)
        return tuple(len(p["prize"]) for p in current["players"])

    def _transition_reward(self, obs):
        now = self._prize_counts(obs)
        old = self._prizes
        self._prizes = now
        mine, theirs = self.learner_slot, 1 - self.learner_slot
        return self.prize_reward * ((old[theirs] - now[theirs]) - (old[mine] - now[mine]))

    def _advance_opponent(self):
        reward = 0.0
        while _result(self.obs) < 0 and self.obs["current"]["yourIndex"] != self.learner_slot:
            action = _safe_rule_action(self.opponent.act, self.obs)
            self.obs = battle_select(action)
            self.engine_decisions += 1
            reward += self._transition_reward(self.obs)
        return reward

    def step(self, action: list[int]):
        if self.obs is None or _result(self.obs) >= 0:
            raise RuntimeError("Call reset() before step()")
        select = self.obs["select"]
        action = [int(x) for x in action]
        valid = (int(select["minCount"]) <= len(action) <= int(select["maxCount"]) and
                 len(set(action)) == len(action) and all(0 <= x < len(select["option"]) for x in action))
        if not valid:
            raise ValueError(f"Illegal learner action {action} for min={select['minCount']} max={select['maxCount']} options={len(select['option'])}")
        self.obs = battle_select(action)
        self.learner_decisions += 1
        self.engine_decisions += 1
        reward = self._transition_reward(self.obs) - self.decision_penalty
        reward += self._advance_opponent()
        winner = _result(self.obs)
        terminated = winner >= 0
        truncated = self.learner_decisions >= self.max_learner_decisions and not terminated
        if terminated:
            reward += 1.0 if winner == self.learner_slot else -1.0
        elif truncated:
            reward -= 0.1
        info = MatchInfo(self.opponent.name, self.learner_slot, None if winner < 0 else winner,
                         self.learner_decisions, self.engine_decisions).__dict__
        return self.obs, float(reward), terminated, truncated, info
