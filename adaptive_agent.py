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
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
from population_sim import PopulationSpeakerEnv, run_episode_tracked
from skeleton_loop import FixedGainAgent

BEST_FIXED_GAIN: float = 1.0


# ============================================================================
# Immutable data structures
# ============================================================================

@dataclass(frozen=True)
class AgentConfig:
    """Immutable configuration for the adaptive G-estimator agent."""
    target: np.ndarray
    gain_prior: float = 0.28
    action_cap: float = 120.0
    lam: float = 0.995
    ridge: float = 1e-3
    explore0: float = 25.0
    explore_decay: float = 0.78
    action_min: float = -200.0
    action_max: float = 200.0
    seed: int = 0
    diagonal_only: bool = False
    k_clamp: Optional[Tuple[float, float]] = None
    hold_len: int = 1
    settle_window: Optional[int] = None
    probe_schedule: Optional[List[np.ndarray]] = None

    @property
    def live(self) -> bool:
        """Live mode is engaged if any robustness feature is requested."""
        return (
            self.diagonal_only
            or self.k_clamp is not None
            or self.hold_len > 1
            or self.probe_schedule is not None
        )


@dataclass(frozen=True)
class AgentState:
    """Immutable snapshot of the agent's estimation and state-machine state."""
    estimated_G_matrix: np.ndarray
    rls_covariance: Tuple[np.ndarray, np.ndarray]
    trial_count: int = 0
    previous_action: Optional[np.ndarray] = None
    previous_observation: Optional[np.ndarray] = None
    phase: str = "control"
    probe_index: int = 0
    probe_data: Tuple[Tuple[np.ndarray, np.ndarray], ...] = ()
    block_action: Optional[np.ndarray] = None
    block_observations: Tuple[np.ndarray, ...] = ()
    pre_block_observation: Optional[np.ndarray] = None
    baseline_observation: Optional[np.ndarray] = None
    speaker_map: Optional[Dict[str, np.ndarray]] = None


def default_state(config: AgentConfig) -> AgentState:
    """Create the initial agent state from a configuration."""
    return AgentState(
        estimated_G_matrix=-config.gain_prior * np.eye(2),
        rls_covariance=(np.eye(2) * 0.5, np.eye(2) * 0.5),
        trial_count=0,
        phase="probe" if config.probe_schedule else "control",
    )


# ============================================================================
# Pure transition functions
# ============================================================================


def rls_update(state: AgentState, lam: float, a: np.ndarray, dy: np.ndarray) -> AgentState:
    """Full-matrix RLS update — returns a new AgentState with updated estimate."""
    new_covariances: List[np.ndarray] = []
    new_G = state.estimated_G_matrix.copy()
    for i in range(2):
        covariance = state.rls_covariance[i].copy()
        covariance_times_action = covariance @ a
        kalman_gain = covariance_times_action / (
            lam + a @ covariance_times_action
        )
        new_G[i] = state.estimated_G_matrix[i] + kalman_gain * (
            dy[i] - state.estimated_G_matrix[i] @ a
        )
        new_covariances.append(
            (covariance - np.outer(kalman_gain, covariance_times_action)) / lam
        )
    return AgentState(
        estimated_G_matrix=new_G,
        rls_covariance=(new_covariances[0], new_covariances[1]),
        trial_count=state.trial_count,
        previous_action=state.previous_action,
        previous_observation=state.previous_observation,
        phase=state.phase,
        probe_index=state.probe_index,
        probe_data=state.probe_data,
        block_action=state.block_action,
        block_observations=state.block_observations,
        pre_block_observation=state.pre_block_observation,
        baseline_observation=state.baseline_observation,
        speaker_map=state.speaker_map,
    )


def rls_update_diag(
    state: AgentState, lam: float, a: np.ndarray, dy: np.ndarray
) -> AgentState:
    """Diagonal-only RLS update — returns a new AgentState with zeroed cross terms."""
    new_G = state.estimated_G_matrix.copy()
    new_covariances: List[np.ndarray] = []
    for i in range(2):
        action_component = float(a[i])
        diagonal_covariance = float(state.rls_covariance[i][i, i])
        denom = lam + action_component * diagonal_covariance * action_component
        kalman_gain = (
            (diagonal_covariance * action_component) / denom if denom != 0.0 else 0.0
        )
        new_G[i, i] = state.estimated_G_matrix[i, i] + kalman_gain * (
            dy[i] - state.estimated_G_matrix[i, i] * action_component
        )
        cov_i = state.rls_covariance[i].copy()
        cov_i[i, i] = (
            diagonal_covariance - kalman_gain * action_component * diagonal_covariance
        ) / lam
        new_covariances.append(cov_i)
    new_G[0, 1] = new_G[1, 0] = 0.0
    return AgentState(
        estimated_G_matrix=new_G,
        rls_covariance=(new_covariances[0], new_covariances[1]),
        trial_count=state.trial_count,
        previous_action=state.previous_action,
        previous_observation=state.previous_observation,
        phase=state.phase,
        probe_index=state.probe_index,
        probe_data=state.probe_data,
        block_action=state.block_action,
        block_observations=state.block_observations,
        pre_block_observation=state.pre_block_observation,
        baseline_observation=state.baseline_observation,
        speaker_map=state.speaker_map,
    )


def apply_k_clamp(
    state: AgentState, k_clamp: Optional[Tuple[float, float]]
) -> AgentState:
    """Clamp diagonal gains to [kmin, kmax] — returns new state or unchanged state."""
    if k_clamp is None:
        return state
    gain_min, gain_max = k_clamp
    new_G = state.estimated_G_matrix.copy()
    for i in range(2):
        gain = -new_G[i, i]
        new_G[i, i] = -float(np.clip(gain, gain_min, gain_max))
    return AgentState(
        estimated_G_matrix=new_G,
        rls_covariance=state.rls_covariance,
        trial_count=state.trial_count,
        previous_action=state.previous_action,
        previous_observation=state.previous_observation,
        phase=state.phase,
        probe_index=state.probe_index,
        probe_data=state.probe_data,
        block_action=state.block_action,
        block_observations=state.block_observations,
        pre_block_observation=state.pre_block_observation,
        baseline_observation=state.baseline_observation,
        speaker_map=state.speaker_map,
    )


def control_action(
    config: AgentConfig, state: AgentState, obs: np.ndarray
) -> np.ndarray:
    """Compute the control action from the current estimate — pure function."""
    target_gap = config.target - np.asarray(obs, dtype=np.float64)
    action = np.linalg.solve(
        state.estimated_G_matrix.T @ state.estimated_G_matrix
        + config.ridge * np.eye(2),
        state.estimated_G_matrix.T @ target_gap,
    )
    action_norm = np.linalg.norm(action)
    if action_norm > config.action_cap:
        action *= config.action_cap / action_norm
    return np.clip(action, config.action_min, config.action_max)


def settled(state: AgentState, settle_window: Optional[int], hold_len: int) -> np.ndarray:
    """Mean of the tail of block_observations — pure function."""
    window_size = settle_window or max(1, hold_len // 2)
    tail = state.block_observations[-window_size:]
    return np.mean(np.vstack(tail), axis=0)


def start_block(
    state: AgentState, config: AgentConfig, ref_obs: np.ndarray
) -> Tuple[AgentState, np.ndarray]:
    """Initialize a new block — returns (new_state, block_action)."""
    if state.phase == "probe":
        assert config.probe_schedule is not None
        block_action = config.probe_schedule[state.probe_index].copy()
    else:
        block_action = control_action(config, state, ref_obs)
    return AgentState(
        estimated_G_matrix=state.estimated_G_matrix,
        rls_covariance=state.rls_covariance,
        trial_count=state.trial_count,
        previous_action=state.previous_action,
        previous_observation=state.previous_observation,
        phase=state.phase,
        probe_index=state.probe_index,
        probe_data=state.probe_data,
        block_action=block_action,
        block_observations=(),
        pre_block_observation=state.pre_block_observation,
        baseline_observation=state.baseline_observation,
        speaker_map=state.speaker_map,
    ), block_action


def fit_probe(
    config: AgentConfig, state: AgentState
) -> AgentState:
    """Least-squares per-axis gain from probe data — returns new state."""
    probe_actions_matrix = np.vstack([p[0] for p in state.probe_data])
    probe_responses_matrix = np.vstack([p[1] for p in state.probe_data])
    gain = np.array([config.gain_prior, config.gain_prior], dtype=float)
    for i in range(2):
        denom = float(np.sum(probe_actions_matrix[:, i] ** 2))
        if denom > 1e-6:
            gain[i] = (
                -float(
                    np.sum(probe_actions_matrix[:, i] * probe_responses_matrix[:, i])
                )
                / denom
            )
    if config.k_clamp is not None:
        gain = np.clip(gain, config.k_clamp[0], config.k_clamp[1])
    new_G = -np.diag(gain)
    baseline = (
        state.baseline_observation
        if state.baseline_observation is not None
        else state.pre_block_observation
    )
    speaker_map: Dict[str, np.ndarray] = {
        "baseline": np.asarray(baseline, float).copy(),
        "k": gain.copy(),
    }
    return AgentState(
        estimated_G_matrix=new_G,
        rls_covariance=(np.eye(2) * 0.5, np.eye(2) * 0.5),
        trial_count=state.trial_count,
        previous_action=state.previous_action,
        previous_observation=state.previous_observation,
        phase=state.phase,
        probe_index=state.probe_index,
        probe_data=state.probe_data,
        block_action=state.block_action,
        block_observations=state.block_observations,
        pre_block_observation=state.pre_block_observation,
        baseline_observation=state.baseline_observation,
        speaker_map=speaker_map,
    )


def act_simple(
    config: AgentConfig, state: AgentState, obs: np.ndarray
) -> Tuple[AgentState, np.ndarray]:
    """Simple (non-live) act transition — returns (new_state, action)."""
    new_state = state
    # If previous action, learn from it
    if state.previous_action is not None:
        dy = obs - state.previous_observation  # type: ignore[operator]
        new_state = rls_update(
            new_state, config.lam, state.previous_action, dy
        )
    # Compute control action
    action = control_action(config, new_state, obs)
    # Add exploration noise (deterministic from seed + trial_count)
    rng = np.random.default_rng(config.seed + state.trial_count)
    exploration_noise = config.explore0 * (config.explore_decay ** state.trial_count)
    action = np.clip(
        action + rng.normal(0, exploration_noise, size=2),
        config.action_min,
        config.action_max,
    )
    new_state = AgentState(
        estimated_G_matrix=new_state.estimated_G_matrix,
        rls_covariance=new_state.rls_covariance,
        trial_count=state.trial_count + 1,
        previous_action=action.copy(),
        previous_observation=obs.copy(),
        phase=new_state.phase,
        probe_index=new_state.probe_index,
        probe_data=new_state.probe_data,
        block_action=new_state.block_action,
        block_observations=new_state.block_observations,
        pre_block_observation=new_state.pre_block_observation,
        baseline_observation=new_state.baseline_observation,
        speaker_map=new_state.speaker_map,
    )
    return new_state, action.astype(np.float32)


def act_live(
    config: AgentConfig, state: AgentState, obs: np.ndarray
) -> Tuple[AgentState, np.ndarray]:
    """Live act transition (probe + ramp-hold + robust RLS) — returns (new_state, action)."""
    # First call: baseline anchor
    if state.pre_block_observation is None:
        new_state = AgentState(
            estimated_G_matrix=state.estimated_G_matrix,
            rls_covariance=state.rls_covariance,
            trial_count=state.trial_count,
            previous_action=state.previous_action,
            previous_observation=state.previous_observation,
            phase=state.phase,
            probe_index=state.probe_index,
            probe_data=state.probe_data,
            block_action=state.block_action,
            block_observations=state.block_observations,
            pre_block_observation=obs.copy(),
            baseline_observation=obs.copy(),
            speaker_map=state.speaker_map,
        )
        new_state, block_action = start_block(new_state, config, obs)
        return new_state, block_action.astype(np.float32)

    # Inside a hold: buffer the observation, keep same perturbation
    new_block_obs = state.block_observations + (obs.copy(),)
    if len(new_block_obs) < config.hold_len:
        assert state.block_action is not None
        buffered_state = AgentState(
            estimated_G_matrix=state.estimated_G_matrix,
            rls_covariance=state.rls_covariance,
            trial_count=state.trial_count,
            previous_action=state.previous_action,
            previous_observation=state.previous_observation,
            phase=state.phase,
            probe_index=state.probe_index,
            probe_data=state.probe_data,
            block_action=state.block_action,
            block_observations=new_block_obs,
            pre_block_observation=state.pre_block_observation,
            baseline_observation=state.baseline_observation,
            speaker_map=state.speaker_map,
        )
        return buffered_state, state.block_action.astype(np.float32)

    # Block complete: measure settled response
    temp_state = AgentState(
        estimated_G_matrix=state.estimated_G_matrix,
        rls_covariance=state.rls_covariance,
        trial_count=state.trial_count,
        previous_action=state.previous_action,
        previous_observation=state.previous_observation,
        phase=state.phase,
        probe_index=state.probe_index,
        probe_data=state.probe_data,
        block_action=state.block_action,
        block_observations=new_block_obs,
        pre_block_observation=state.pre_block_observation,
        baseline_observation=state.baseline_observation,
        speaker_map=state.speaker_map,
    )
    s = settled(temp_state, config.settle_window, config.hold_len)
    dy = s - state.pre_block_observation  # type: ignore[operator]

    if state.phase == "probe":
        assert state.block_action is not None
        new_probe_data = state.probe_data + ((state.block_action.copy(), dy.copy()),)
        new_state = AgentState(
            estimated_G_matrix=temp_state.estimated_G_matrix,
            rls_covariance=temp_state.rls_covariance,
            trial_count=temp_state.trial_count,
            previous_action=temp_state.previous_action,
            previous_observation=temp_state.previous_observation,
            phase=temp_state.phase,
            probe_index=temp_state.probe_index + 1,
            probe_data=new_probe_data,
            block_action=temp_state.block_action,
            block_observations=temp_state.block_observations,
            pre_block_observation=s.copy(),
            baseline_observation=temp_state.baseline_observation,
            speaker_map=temp_state.speaker_map,
        )
        assert config.probe_schedule is not None
        if new_state.probe_index >= len(config.probe_schedule):
            new_state = fit_probe(config, new_state)
            new_state = AgentState(
                estimated_G_matrix=new_state.estimated_G_matrix,
                rls_covariance=new_state.rls_covariance,
                trial_count=new_state.trial_count,
                previous_action=new_state.previous_action,
                previous_observation=new_state.previous_observation,
                phase="control",
                probe_index=new_state.probe_index,
                probe_data=new_state.probe_data,
                block_action=new_state.block_action,
                block_observations=new_state.block_observations,
                pre_block_observation=new_state.pre_block_observation,
                baseline_observation=new_state.baseline_observation,
                speaker_map=new_state.speaker_map,
            )
        new_state, block_action = start_block(new_state, config, s)
        return new_state, block_action.astype(np.float32)

    # Control phase: robust RLS on settled block response
    assert state.block_action is not None
    if config.diagonal_only:
        new_state = rls_update_diag(
            temp_state, config.lam, state.block_action, dy
        )
    else:
        new_state = rls_update(
            temp_state, config.lam, state.block_action, dy
        )
    new_state = apply_k_clamp(new_state, config.k_clamp)
    new_state = AgentState(
        estimated_G_matrix=new_state.estimated_G_matrix,
        rls_covariance=new_state.rls_covariance,
        trial_count=new_state.trial_count,
        previous_action=new_state.previous_action,
        previous_observation=new_state.previous_observation,
        phase=new_state.phase,
        probe_index=new_state.probe_index,
        probe_data=new_state.probe_data,
        block_action=new_state.block_action,
        block_observations=new_state.block_observations,
        pre_block_observation=s.copy(),
        baseline_observation=new_state.baseline_observation,
        speaker_map=new_state.speaker_map,
    )
    new_state, block_action = start_block(new_state, config, s)
    return new_state, block_action.astype(np.float32)


# ============================================================================
# Probe schedule factory
# ============================================================================

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


# ============================================================================
# Convenience wrapper — thin facade over pure transition functions
# ============================================================================

class AdaptiveAgent:
    """Thin wrapper that holds immutable config + state and delegates to pure
    transition functions.  Preserves the familiar API (agent.act(obs), etc.)
    while the actual logic is purely functional."""

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
        probe_schedule: Optional[
            Sequence[Union[np.ndarray, Sequence[float]]]
        ] = None,
    ):
        # Build immutable config
        sched_list: Optional[List[np.ndarray]] = None
        if probe_schedule is not None:
            sched_list = [np.asarray(p, float) for p in probe_schedule]
        self.config = AgentConfig(
            target=np.asarray(target, dtype=np.float64),
            gain_prior=float(gain_prior),
            action_cap=float(action_cap),
            lam=float(lam),
            ridge=float(ridge),
            explore0=float(explore0),
            explore_decay=float(explore_decay),
            action_min=float(action_low),
            action_max=float(action_high),
            seed=int(seed),
            diagonal_only=bool(diagonal_only),
            k_clamp=(float(k_clamp[0]), float(k_clamp[1])) if k_clamp is not None else None,
            hold_len=max(1, int(hold_len)),
            settle_window=int(settle_window) if settle_window is not None else None,
            probe_schedule=sched_list,
        )
        # Initialize immutable state
        self.state = default_state(self.config)

    # --- property accessors (delegate to state) ---
    @property
    def target(self) -> np.ndarray:
        return self.config.target

    @property
    def estimated_G_matrix(self) -> np.ndarray:
        return self.state.estimated_G_matrix

    @property
    def rls_covariance(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.state.rls_covariance

    @property
    def trial_count(self) -> int:
        return self.state.trial_count

    @property
    def previous_action(self) -> Optional[np.ndarray]:
        return self.state.previous_action

    @property
    def previous_observation(self) -> Optional[np.ndarray]:
        return self.state.previous_observation

    @property
    def phase(self) -> str:
        return self.state.phase

    @property
    def probe_index(self) -> int:
        return self.state.probe_index

    @property
    def probe_data(self) -> Tuple[Tuple[np.ndarray, np.ndarray], ...]:
        return self.state.probe_data

    @property
    def block_action(self) -> Optional[np.ndarray]:
        return self.state.block_action

    @property
    def block_observations(self) -> Tuple[np.ndarray, ...]:
        return self.state.block_observations

    @property
    def pre_block_observation(self) -> Optional[np.ndarray]:
        return self.state.pre_block_observation

    @property
    def baseline_observation(self) -> Optional[np.ndarray]:
        return self.state.baseline_observation

    @property
    def speaker_map(self) -> Optional[Dict[str, np.ndarray]]:
        return self.state.speaker_map

    @property
    def live(self) -> bool:
        return self.config.live

    @property
    def _live(self) -> bool:
        return self.config.live

    @property
    def gain_prior(self) -> float:
        return self.config.gain_prior

    @property
    def diagonal_only(self) -> bool:
        return self.config.diagonal_only

    @property
    def k_clamp(self) -> Optional[Tuple[float, float]]:
        return self.config.k_clamp

    @property
    def hold_len(self) -> int:
        return self.config.hold_len

    @property
    def settle_window(self) -> Optional[int]:
        return self.config.settle_window

    @property
    def probe_schedule(self) -> Optional[List[np.ndarray]]:
        return self.config.probe_schedule

    @property
    def action_cap(self) -> float:
        return self.config.action_cap

    @property
    def action_min(self) -> float:
        return self.config.action_min

    @property
    def action_max(self) -> float:
        return self.config.action_max

    @property
    def exploration_noise_init(self) -> float:
        return self.config.explore0

    @property
    def exploration_noise_decay(self) -> float:
        return self.config.explore_decay

    # --- public API ---
    def act(self, observation: Union[np.ndarray, Sequence[float]]) -> np.ndarray:
        obs = np.asarray(observation, dtype=np.float64)
        if not self.config.live:
            self.state, action = act_simple(self.config, self.state, obs)
        else:
            self.state, action = act_live(self.config, self.state, obs)
        return action

    def reset_block_state(self, keep_estimate: bool = True) -> None:
        """Start a fresh session. Keeps the learned estimate / probe map (warm
        start); only the within-session block bookkeeping and anchor are cleared.
        If a probe already ran, phase stays 'control' so the next session does
        NOT re-probe."""
        if keep_estimate:
            self.state = AgentState(
                estimated_G_matrix=self.state.estimated_G_matrix,
                rls_covariance=self.state.rls_covariance,
                trial_count=0,
                previous_action=None,
                previous_observation=None,
                phase=self.state.phase,
                probe_index=self.state.probe_index,
                probe_data=self.state.probe_data,
                block_action=None,
                block_observations=(),
                pre_block_observation=None,
                baseline_observation=None,
                speaker_map=self.state.speaker_map,
            )
        else:
            self.state = default_state(self.config)


# Backward-compatible alias
AdaptiveGEstimatorAgent = AdaptiveAgent


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
