"""
marl_layer.py

Wraps your real defender_final.npy checkpoint and a faithful
reimplementation of its training environment (MASEnv from
AAR_FIXED_FINAL_.ipynb). This is kept as a SEPARATE layer from the text
classifier in detection_engine.py, because that's what your notebook's
own architecture does: aar_framework.py's process() decides block/allow
purely from AgentArmon + IPIGuard + the Collaborative LLM vote. The MARL
Q-network's job is graph-level agent/device isolation, not reading
message text - it never receives the message as input.

Import note: numpy is required; this loads the .npy checkpoint directly,
exactly as your notebook exports it (no JSON conversion).
"""

import numpy as np

ACTIONS = ["process", "quarantine", "verify"]


class QNet:
    def __init__(self, weights):
        self.W1 = weights["W1"]
        self.b1 = weights["b1"]
        self.W2 = weights["W2"]
        self.b2 = weights["b2"]

    def forward(self, x):
        h = np.maximum(0.0, x @ self.W1 + self.b1)
        return h @ self.W2 + self.b2


def load_defender(npy_path: str) -> QNet:
    weights = np.load(npy_path, allow_pickle=True).item()
    return QNet(weights)


class GraphEnv:
    """Faithful port of MASEnv, same as iot_graph_env.IoTGraphEnv."""

    def __init__(self, n=5, max_steps=200, seed=None):
        self.n = n
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        self.t = 0
        self.isolated = set()
        self.compromised_idx = int(self.rng.integers(0, self.n))
        self.trust = np.full(self.n, 0.5)
        self.consensus = np.zeros(self.n)
        self.task = self._new_task()
        self.injection = False
        self.atk_attempts = 0
        self.blocked = 0
        self.legit = 0
        self.fp = 0

    def _new_task(self):
        return self.rng.uniform(-1, 1, self.n + 4)

    def obs(self):
        v = np.concatenate([self.task[:4], self.trust, self.consensus, [float(self.injection)]])
        return np.clip(v, -1, 1)

    def _attacker_policy(self):
        if self.compromised_idx in self.isolated:
            return 0
        mean = self.trust.mean()
        return 2 if mean > 0.7 else (1 if mean > 0.4 else 0)

    def _inject(self, strength):
        for nb in range(self.n):
            if nb == self.compromised_idx or nb in self.isolated:
                continue
            self.consensus[nb] = (1 - strength) * self.consensus[nb] + strength
            self.trust[nb] = max(0.0, self.trust[nb] - 0.06)

    def _most_suspicious(self):
        mean = self.consensus.mean()
        dev = np.abs(self.consensus - mean)
        for i in self.isolated:
            dev[i] = -1
        return int(np.argmax(dev))

    def _isolate(self, i):
        self.isolated.add(i)
        self.trust[i] = 0.0

    def step(self, action: int, force_attack: bool = False):
        self.t += 1
        reward = 0.0
        if force_attack and self.compromised_idx not in self.isolated:
            # Demo coupling: the text filter's verdict forces an attack
            # attempt this tick, rather than relying only on the graph's
            # own internal attacker schedule. This is an addition for
            # this app - your original MASEnv/IoTGraphEnv has no such
            # hook, since the MARL layer was never meant to react to
            # text-filter verdicts.
            self.atk_attempts += 1
            self.injection = True
            self._inject(0.7)
        else:
            atk = self._attacker_policy()
            if atk == 1:
                self.atk_attempts += 1
                self.injection = True
                self._inject(0.3)
            elif atk == 2:
                self.atk_attempts += 1
                self.injection = True
                self._inject(0.7)

        if action == 0:
            if not self.injection:
                self.legit += 1
                reward += 8
            else:
                reward -= 5
        elif action == 1:
            f = self._most_suspicious()
            if f == self.compromised_idx and f not in self.isolated:
                self._isolate(f)
                self.blocked += 1
                reward += 25
                self.injection = False
            else:
                self.fp += 1
                reward -= 3
        elif action == 2:
            reward += 5 if self.injection else -1
            reward -= 3

        self.task = self._new_task()
        if self.t >= self.max_steps:
            self.reset()
        return reward


def tick(defender: QNet, env: GraphEnv, force_attack: bool = False):
    """Runs one step of the network defense layer, returns the action taken."""
    obs = env.obs()
    q = defender.forward(obs)
    action_idx = int(np.argmax(q))
    env.step(action_idx, force_attack=force_attack)
    return ACTIONS[action_idx]
