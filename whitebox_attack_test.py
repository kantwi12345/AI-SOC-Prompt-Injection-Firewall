"""
whitebox_attack_test.py

Tests both the original and retrained MARL defenders against a
white-box adaptive attacker that has full knowledge of the defender's
exact Q-network weights - the direct analog of MedSentry's
"attacker-knows-the-defense-rules" experiment (their Appendix B.2).

The environment's built-in attacker (_attacker_policy in GraphEnv) is a
FIXED script that doesn't react to the defender at all - it's the
"attacker doesn't know the rules" baseline. This script replaces it
with an adaptive attacker that, at each step, evaluates several
candidate injection strengths, computes what the defender's Q-network
would output for the resulting observation, and picks the strength
that minimizes the defender's Q-value for "isolate" - i.e., the
strategy that best evades detection while still injecting.

This is a genuine test, not a guaranteed result either way - reported
honestly regardless of outcome.
"""

import numpy as np
from marl_layer import QNet, load_defender


class WhiteboxAdaptiveEnv:
    """Same core mechanics as GraphEnv, but the attacker has white-box
    access to the defender's Q-network and picks its injection strength
    adaptively each step to minimize the defender's isolate-action Q-value."""

    def __init__(self, defender: QNet, n=5, max_steps=1000, seed=None):
        self.defender = defender
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

    def _simulate_inject(self, strength):
        """Returns what trust/consensus WOULD be after injecting at this
        strength, without committing the change - used by the adaptive
        attacker to evaluate candidates."""
        trust = self.trust.copy()
        consensus = self.consensus.copy()
        for nb in range(self.n):
            if nb == self.compromised_idx or nb in self.isolated:
                continue
            consensus[nb] = (1 - strength) * consensus[nb] + strength
            trust[nb] = max(0.0, trust[nb] - 0.06)
        return trust, consensus

    def _adaptive_attacker_strength(self):
        """White-box: try candidate strengths, pick the one that
        minimizes the defender's Q-value for action=1 (isolate), i.e.
        the strength least likely to get the compromised device flagged,
        while still being > 0 (the attacker still wants to inject)."""
        if self.compromised_idx in self.isolated:
            return 0.0
        candidates = [0.05, 0.10, 0.15, 0.20, 0.30, 0.45, 0.65]
        best_strength, best_isolate_q = 0.0, float("inf")
        for s in candidates:
            trust, consensus = self._simulate_inject(s)
            hypothetical_obs = np.clip(
                np.concatenate([self.task[:4], trust, consensus, [1.0]]), -1, 1
            )
            q = self.defender.forward(hypothetical_obs)
            isolate_q = q[1]
            if isolate_q < best_isolate_q:
                best_isolate_q = isolate_q
                best_strength = s
        return best_strength

    def _most_suspicious(self):
        mean = self.consensus.mean()
        dev = np.abs(self.consensus - mean)
        for i in self.isolated:
            dev[i] = -1
        return int(np.argmax(dev))

    def _isolate(self, i):
        self.isolated.add(i)
        self.trust[i] = 0.0

    def _inject(self, strength):
        for nb in range(self.n):
            if nb == self.compromised_idx or nb in self.isolated:
                continue
            self.consensus[nb] = (1 - strength) * self.consensus[nb] + strength
            self.trust[nb] = max(0.0, self.trust[nb] - 0.06)

    def step(self, action: int):
        self.t += 1
        reward = 0.0

        strength = self._adaptive_attacker_strength()
        if strength > 0:
            self.atk_attempts += 1
            self.injection = True
            self._inject(strength)
        else:
            self.injection = False

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


def evaluate_whitebox(npy_path, n_episodes=50, steps_per_episode=80, seed=7):
    defender = load_defender(npy_path)
    rng = np.random.default_rng(seed)
    per_ep_asr, per_ep_blocked, per_ep_fp = [], [], []
    for ep in range(n_episodes):
        env = WhiteboxAdaptiveEnv(defender, seed=int(rng.integers(0, 1_000_000)),
                                    max_steps=steps_per_episode + 1000)
        for _ in range(steps_per_episode):
            obs = env.obs()
            action = int(defender.forward(obs).argmax())
            env.step(action)
        asr = (env.atk_attempts - env.blocked) / max(1, env.atk_attempts)
        per_ep_asr.append(asr)
        per_ep_blocked.append(env.blocked)
        per_ep_fp.append(env.fp)
    return {
        "detection_rate": 1 - np.mean(per_ep_asr),
        "blocked_per_ep": np.mean(per_ep_blocked),
        "fp_per_ep": np.mean(per_ep_fp),
    }


if __name__ == "__main__":
    print("Testing both defenders against a white-box adaptive attacker")
    print("(full knowledge of the defender's Q-network, picks injection")
    print("strength each step specifically to evade the 'isolate' action)\n")

    for label, path in [("Original defender", "defender_final_original.npy"),
                          ("Retrained defender", "defender_retrained.npy")]:
        result = evaluate_whitebox(path)
        print(f"=== {label} ===")
        print(f"  Detection rate:          {result['detection_rate']*100:.1f}%")
        print(f"  Quarantined per episode: {result['blocked_per_ep']:.2f}")
        print(f"  False positives/episode: {result['fp_per_ep']:.2f}")
        print()
