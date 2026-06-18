"""
adaptive_agent.py
========================================
Adaptive G-estimator agent on the same 60 seeded speakers from population_sim.py.

The adaptive agent's job is to estimate each speaker's response
matrix G online via RLS and act by inverting the estimate.

LIVE-SESSION ROBUSTNESS (added)
-------------------------------
Three features were added for the real Audapter loop. They are ALL opt-in and
default OFF, so the simulation path (population_sim / multi_session_robustness)
is byte-for-byte unchanged:

  (2) ramp-and-hold:   hold_len > 1 holds each perturbation for several trials and
                       updates the estimate on the *settled* block response, not the
                       single-trial delta. Real auditory-motor compensation is a slow
                       integrator; a memoryless per-trial delta is mostly noise.
  (3) robust estimator: diagonal_only=True drops the off-diagonal cross terms (the
                       first thing to diverge cold), and k_clamp=(kmin,kmax) bounds
                       the per-axis gain so one bad take can't send k to -2.
  (5) probe warm-start: probe_schedule runs a short scripted sequence of held,
                       axis-aligned perturbations FIRST, identifies each speaker's
                       gains cleanly, warm-starts W, then switches to closed-loop
                       control. The probe also yields self.speaker_map -- the natural
                       hook for normalizing targets into per-speaker space.

IMPORTANT NUMBERS

gain_mean(F1): 0.26 per [1]
gain_mean(F2): 0.30 per [1]
gain_std: 0.1-0.15 [1][2]
cross_coupling_std: 0.03-0.05 [1][3][4]
saturation_amount: 70 Hz [1][5]
detection_threshold: 200-300 Hz [2]
noise_std: 10-20 Hz
max_steps: 20-40 (per phase) [1][3]

[1] MacDonald
[2] House & Jordan
[3] Purcell
[4] Munhall
[5] Katseff
"""

import matplotlib
import numpy as np

matplotlib.use("Agg")
from typing import Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
from population_sim import PopulationSpeakerEnv, run_episode_tracked
from skeleton_loop import FixedGainAgent

BEST_FIXED_GAIN: float = 1.0


def make_probe_schedule(probe_amp: float = 80.0, n_axes: int = 2) -> List[np.ndarray]:
    """Axis-aligned probe perturbations for cold per-speaker identification.

    Each probe vector is held by the agent for `hold_len` trials; the settled
    compensation across the hold identifies that axis's gain. The default 2-axis
    schedule [(P,0),(0,P)] is the minimum to read k_F1 and k_F2 separately.
    Add +/- pairs (drift cancellation) or extra amplitudes if your trial budget
    allows -- each extra probe costs hold_len trials.
    """
    P = float(probe_amp)
    sched = [
        np.array([P, 0.0]),
        np.array([0.0, P]),
        np.array([-P, 0.0]),
        np.array([0.0, -P]),
    ]
    return sched[: max(1, int(n_axes))]


class AdaptiveGEstimatorAgent:
    # initializing agent
    def __init__(
        self,
        target: Union[np.ndarray, Sequence[float]],
        gain_prior: float = 0.28,
        action_cap: float = 120.0,
        lam: float = 0.995,
        ridge: float = 1e-3,
        explore0: float = 25.0,
        explore_decay: float = 0.78,
        action_low: float = -200.0,
        action_high: float = 200.0,
        seed: int = 0,
        # --- live-session robustness; defaults OFF => identical sim behavior ---
        diagonal_only: bool = False,
        k_clamp: Optional[Tuple[float, float]] = None,
        hold_len: int = 1,
        settle_window: Optional[int] = None,
        probe_schedule: Optional[Sequence[Union[np.ndarray, Sequence[float]]]] = None,
    ):
        self.target = np.asarray(target, dtype=np.float64)
        # Prior G = population-mean diagonal (avg of MacDonald 2010's k_F1, k_F2)
        self.estimated_G_matrix = -gain_prior * np.eye(2)  # agent estimate of G
        self.rls_covariance = [np.eye(2) * 0.5, np.eye(2) * 0.5]  # RLS covariance
        self.forgetting_factor, self.ridge = (
            lam,
            ridge,
        )  # RLS forgetting factor (long bc lam close to 1)
        self.action_cap = action_cap  # Max norm for control action
        self.exploration_noise_init, self.exploration_noise_decay = (
            explore0,
            explore_decay,
        )  # exploration noise
        self.action_min, self.action_max = action_low, action_high
        self.rng = np.random.default_rng(seed)
        self.previous_action = self.previous_observation = (
            None  # history, meaningful when multiple sessions
        )
        self.trial_count = 0
        self.gain_prior = float(gain_prior)

        # --- live-session features ---
        self.diagonal_only = bool(diagonal_only)
        self.k_clamp = tuple(k_clamp) if k_clamp is not None else None
        self.hold_len = max(1, int(hold_len))
        self.settle_window = settle_window  # None -> back half of the hold
        self.probe_schedule = (
            [np.asarray(p, float) for p in probe_schedule] if probe_schedule else None
        )
        # live mode engages if any robustness feature is requested
        self._live = (
            self.diagonal_only
            or self.k_clamp is not None
            or self.hold_len > 1
            or self.probe_schedule is not None
        )

        # live state machine
        self.phase: str = "probe" if self.probe_schedule else "control"
        self.probe_index: int = 0
        self.probe_data: List[
            Tuple[np.ndarray, np.ndarray]
        ] = []  # list of (held_action, settled_response)
        self.block_action: Optional[np.ndarray] = None
        self.block_observations: List[np.ndarray] = []
        self.pre_block_observation: Optional[np.ndarray] = (
            None  # settled production at the start of this block
        )
        self.baseline_observation: Optional[np.ndarray] = (
            None  # the very first (baseline) observation
        )
        self.speaker_map: Optional[Dict[str, np.ndarray]] = (
            None  # {"baseline": [F1,F2], "k": [kF1,kF2]} after probe
        )

    # ===================== estimator updates =====================
    def _rls_update(self, a: np.ndarray, dy: np.ndarray) -> None:
        # ORIGINAL full-matrix RLS (unchanged) -- the simulation path uses this.
        for i in range(2):
            covariance = self.rls_covariance[i]
            covariance_times_action = covariance @ a
            # update filter coefficients
            kalman_gain = covariance_times_action / (
                self.forgetting_factor + a @ covariance_times_action
            )
            self.estimated_G_matrix[i] = self.estimated_G_matrix[i] + kalman_gain * (
                dy[i] - self.estimated_G_matrix[i] @ a
            )
            self.rls_covariance[i] = (
                covariance - np.outer(kalman_gain, covariance_times_action)
            ) / self.forgetting_factor

    def _rls_update_diag(self, a: np.ndarray, dy: np.ndarray) -> None:
        # Two independent scalar RLS filters: dy[i] ~ estimated_G_matrix[i,i] * a[i]. No cross terms,
        # so the off-diagonal can never absorb noise and blow up.
        for i in range(2):
            action_component = float(a[i])
            diagonal_covariance = float(self.rls_covariance[i][i, i])
            denom = (
                self.forgetting_factor
                + action_component * diagonal_covariance * action_component
            )
            kalman_gain = (
                (diagonal_covariance * action_component) / denom
                if denom != 0.0
                else 0.0
            )
            self.estimated_G_matrix[i, i] = self.estimated_G_matrix[
                i, i
            ] + kalman_gain * (dy[i] - self.estimated_G_matrix[i, i] * action_component)
            self.rls_covariance[i][i, i] = (
                diagonal_covariance
                - kalman_gain * action_component * diagonal_covariance
            ) / self.forgetting_factor
        self.estimated_G_matrix[0, 1] = self.estimated_G_matrix[1, 0] = 0.0

    def _apply_k_clamp(self) -> None:
        if self.k_clamp is None:
            return
        gain_min, gain_max = self.k_clamp
        for i in range(2):
            gain = -self.estimated_G_matrix[i, i]
            self.estimated_G_matrix[i, i] = -float(np.clip(gain, gain_min, gain_max))

    # ===================== control law =====================
    def _control_action(self, obs: np.ndarray) -> np.ndarray:
        target_gap = self.target - np.asarray(obs, dtype=np.float64)  # gap to target
        # Solve, then perturb opposite (estimated_G_matrix ~ -k*I  =>  action ~ -target_gap/k)
        action = np.linalg.solve(
            self.estimated_G_matrix.T @ self.estimated_G_matrix
            + self.ridge * np.eye(2),
            self.estimated_G_matrix.T @ target_gap,
        )
        action_norm = np.linalg.norm(action)
        if action_norm > self.action_cap:
            action *= self.action_cap / action_norm
        return np.clip(action, self.action_min, self.action_max)

    # ===================== public act =====================
    def act(self, observation: Union[np.ndarray, Sequence[float]]) -> np.ndarray:
        obs = np.asarray(observation, dtype=np.float64)
        if not self._live:
            return self._act_simple(obs)
        return self._act_live(obs)

    def _act_simple(self, obs: np.ndarray) -> np.ndarray:
        # EXACT original behavior (simulation path) -- do not modify.
        # if previous action, learn from it
        if self.previous_action is not None:
            self._rls_update(self.previous_action, obs - self.previous_observation)
        target_gap = self.target - obs  # learn from gap to target
        # Solves and then perturbs opposite (estimated_G_matrix ~ -0.28*I thus action ~ -3.6*target_gap)
        action = np.linalg.solve(
            self.estimated_G_matrix.T @ self.estimated_G_matrix
            + self.ridge * np.eye(2),
            self.estimated_G_matrix.T @ target_gap,
        )
        action_norm = np.linalg.norm(action)
        if action_norm > self.action_cap:
            action *= self.action_cap / action_norm
        # Explores and adds some noise
        exploration_noise = self.exploration_noise_init * (
            self.exploration_noise_decay**self.trial_count
        )
        action = np.clip(
            action + self.rng.normal(0, exploration_noise, size=2),
            self.action_min,
            self.action_max,
        )
        self.previous_action, self.previous_observation = action.copy(), obs.copy()
        self.trial_count += 1
        return action.astype(np.float32)

    # ===================== live: probe + ramp-hold + robust RLS =====================
    def _settled(self) -> np.ndarray:
        """Settled production over the back of the hold (compensation has built)."""
        window_size = self.settle_window or max(1, self.hold_len // 2)
        tail = self.block_observations[-window_size:]
        return np.mean(np.vstack(tail), axis=0)

    def _start_block(self, ref_obs: np.ndarray) -> np.ndarray:
        if self.phase == "probe":
            assert self.probe_schedule is not None
            self.block_action = self.probe_schedule[self.probe_index].copy()
        else:
            self.block_action = self._control_action(ref_obs)
        self.block_observations = []
        return self.block_action

    def _fit_probe(self) -> None:
        """Least-squares per-axis gain from held probe responses: dy ~ -k * a."""
        probe_actions_matrix = np.vstack(
            [p[0] for p in self.probe_data]
        )  # actions  [n,2]
        probe_responses_matrix = np.vstack(
            [p[1] for p in self.probe_data]
        )  # responses [n,2]
        gain = np.array([self.gain_prior, self.gain_prior], dtype=float)
        for i in range(2):
            denom = float(np.sum(probe_actions_matrix[:, i] ** 2))
            if denom > 1e-6:
                gain[i] = (
                    -float(
                        np.sum(
                            probe_actions_matrix[:, i] * probe_responses_matrix[:, i]
                        )
                    )
                    / denom
                )
        if self.k_clamp is not None:
            gain = np.clip(gain, self.k_clamp[0], self.k_clamp[1])
        self.estimated_G_matrix = -np.diag(gain)
        self.rls_covariance = [
            np.eye(2) * 0.5,
            np.eye(2) * 0.5,
        ]  # fresh covariance for control
        baseline = (
            self.baseline_observation
            if self.baseline_observation is not None
            else self.pre_block_observation
        )
        self.speaker_map = {
            "baseline": np.asarray(baseline, float).copy(),
            "k": gain.copy(),
        }

    def _act_live(self, obs: np.ndarray) -> np.ndarray:
        # First call is the baseline anchor MATLAB hands over: set the reference
        # and emit the first block's action (no buffering, no update yet).
        if self.pre_block_observation is None:
            self.baseline_observation = obs.copy()
            self.pre_block_observation = obs.copy()
            return self._start_block(obs).astype(np.float32)

        # Inside a hold: buffer the valid take, keep the SAME perturbation.
        self.block_observations.append(obs.copy())
        if len(self.block_observations) < self.hold_len:
            assert self.block_action is not None
            return self.block_action.astype(np.float32)

        # Block complete: measure settled response to the held action.
        settled = self._settled()
        dy = settled - self.pre_block_observation

        if self.phase == "probe":
            assert self.block_action is not None
            self.probe_data.append((self.block_action.copy(), dy.copy()))
            self.probe_index += 1
            self.pre_block_observation = settled.copy()
            assert self.probe_schedule is not None
            if self.probe_index >= len(self.probe_schedule):
                self._fit_probe()  # warm-start estimated_G_matrix from the probe, switch phases
                self.phase = "control"
            return self._start_block(settled).astype(np.float32)

        # control phase: robust RLS on the settled block response, then re-aim
        assert self.block_action is not None
        if self.diagonal_only:
            self._rls_update_diag(self.block_action, dy)
        else:
            self._rls_update(self.block_action, dy)
        self._apply_k_clamp()
        self.pre_block_observation = settled.copy()
        return self._start_block(settled).astype(np.float32)

    # ---- live carryover control (used by the bridge between sessions) ----
    def reset_block_state(self, keep_estimate: bool = True) -> None:
        """Start a fresh session. Keeps the learned estimated_G_matrix / speaker_map (warm start);
        only the within-session block bookkeeping and anchor are cleared. If a probe
        already ran, phase stays 'control' so the next session does NOT re-probe."""
        self.previous_action = self.previous_observation = None
        self.block_action = None
        self.block_observations = []
        self.pre_block_observation = None
        self.baseline_observation = None
        if not keep_estimate:
            self.estimated_G_matrix = -self.gain_prior * np.eye(2)
            self.rls_covariance = [np.eye(2) * 0.5, np.eye(2) * 0.5]
            self.phase = "probe" if self.probe_schedule else "control"
            self.probe_index = 0
            self.probe_data = []
            self.speaker_map = None


if __name__ == "__main__":
    num_speakers, MAX_STEPS = 60, 30  # 60 simulated speakers, 30 trials each
    speaker_seeds = list(range(num_speakers))
    env = PopulationSpeakerEnv(max_steps=MAX_STEPS, seed=0)
    fixed = FixedGainAgent(target=env.target, gain=BEST_FIXED_GAIN)

    fixed_distance_curves: List[List[float]] = []
    fixed_final_vals: List[float] = []
    speaker_G_matrices: List[np.ndarray] = []
    adaptive_distance_curves: List[List[float]] = []
    adaptive_final_vals: List[float] = []

    # for each seed s, runs once with the fixed and once with the adaptive
    for s in speaker_seeds:
        df, G = run_episode_tracked(env, fixed, seed=s)
        fixed_distance_curves.append(df)
        fixed_final_vals.append(df[-1])
        speaker_G_matrices.append(G)
        ad = AdaptiveGEstimatorAgent(target=env.target, seed=1000 + s)
        da, _ = run_episode_tracked(env, ad, seed=s)
        adaptive_distance_curves.append(da)
        adaptive_final_vals.append(da[-1])

    fixed_final_arr = np.array(fixed_final_vals)
    adaptive_final_arr = np.array(adaptive_final_vals)

    # Prinout/Metrics
    speaker_G_arr = np.array(speaker_G_matrices)
    cross_coupling_magnitude = np.sqrt(
        speaker_G_arr[:, 0, 1] ** 2 + speaker_G_arr[:, 1, 0] ** 2
    )
    anisotropy = np.abs(-speaker_G_arr[:, 0, 0] - (-speaker_G_arr[:, 1, 1]))

    fixed_converged, fixed_failed = fixed_final_arr < 15, fixed_final_arr > 50
    adaptive_converged, adaptive_failed = adaptive_final_arr < 15, adaptive_final_arr > 50

    print(
        f"Paired comparison on {num_speakers} identical literature-grounded speakers, "
        f"{MAX_STEPS} trials:\n"
    )
    print(
        f"  Fixed (gain {BEST_FIXED_GAIN}):  conv {100 * fixed_converged.mean():.0f}%   "
        f"fail {100 * fixed_failed.mean():.0f}%   mean {fixed_final_arr.mean():.1f}   "
        f"median {np.median(fixed_final_arr):.1f}"
    )
    print(
        f"  Adaptive:          conv {100 * adaptive_converged.mean():.0f}%   "
        f"fail {100 * adaptive_failed.mean():.0f}%   mean {adaptive_final_arr.mean():.1f}   "
        f"median {np.median(adaptive_final_arr):.1f}"
    )

    speakers_rescued_by_adaptive = fixed_failed & ~adaptive_failed
    print(
        f"\n  Speakers fixed fail: {fixed_failed.sum()}; adaptive saved: {speakers_rescued_by_adaptive.sum()}"
    )
    if fixed_failed.sum() > 0:
        print(
            f"  Fixed-fail tail final dist -- fixed mean {fixed_final_arr[fixed_failed].mean():.1f}"
            f"  vs adaptive mean {adaptive_final_arr[fixed_failed].mean():.1f}"
        )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    fixed_mean_curve = np.mean(np.vstack(fixed_distance_curves), axis=0)
    adaptive_mean_curve = np.mean(np.vstack(adaptive_distance_curves), axis=0)
    ax1.plot(
        fixed_mean_curve,
        "o-",
        color="tab:red",
        lw=2,
        label=f"Fixed gain {BEST_FIXED_GAIN}",
    )
    ax1.plot(
        adaptive_mean_curve,
        "o-",
        color="tab:blue",
        lw=2,
        label="Adaptive (G-estimator)",
    )
    ax1.axhline(15, color="black", lw=0.6, ls="--")
    ax1.set_xlabel("Trial")
    ax1.set_ylabel("Mean true distance to /u/ (60 speakers)")
    ax1.set_title("Average convergence: adaptive vs fixed (literature-grounded)")
    ax1.grid(alpha=0.3)
    ax1.legend(loc="best")

    scatter_collection = ax2.scatter(
        fixed_final_arr,
        adaptive_final_arr,
        c=anisotropy,
        cmap="viridis",
        s=70,
        edgecolor="black",
        lw=0.5,
    )
    axis_limit = max(fixed_final_arr.max(), adaptive_final_arr.max(), 50) * 1.05
    ax2.plot([0, axis_limit], [0, axis_limit], "k--", lw=0.8)
    ax2.axhline(15, color="gray", lw=0.5)
    ax2.axvline(50, color="gray", lw=0.5)
    ax2.set_xlim(0, axis_limit)
    ax2.set_ylim(0, axis_limit)
    ax2.set_xlabel("Fixed-agent final distance")
    ax2.set_ylabel("Adaptive-agent final distance")
    ax2.set_title("Per-speaker: points below diagonal = adaptive wins")
    ax2.grid(alpha=0.3)
    fig.colorbar(  # type: ignore[unknown-member]
        scatter_collection, ax=ax2, label="gain anisotropy |k$_{F1}$-k$_{F2}$|"
    )

    fig.tight_layout()
    fig.savefig("adaptive_vs_fixed.png", dpi=150)  # type: ignore[unknown-member]
    print("\nWrote: adaptive_vs_fixed.png")
