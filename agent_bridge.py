"""
agent_bridge.py
===============
Thin bridge so MATLAB (AdaptationRun.m) can drive the Python AdaptiveAgent
between trials. MATLAB calls these module-level functions via the py.* interface.

The interface deliberately uses plain scalars / lists (no numpy objects cross the
MATLAB <-> Python boundary) so data passing stays robust across MATLAB releases.

Lifecycle from MATLAB:
    py.agent_bridge.make_agent(target_f1, target_f2, seed)   % once, before the loop
    py.agent_bridge.reset_session()                          % at the start of each session
    action = py.agent_bridge.act(obs_f1, obs_f2)             % after each trial -> next perturbation

LIVE ROBUSTNESS (items 2/3/5) is ON by default here. The agent now:
  * runs a short HELD probe block first to identify the speaker's gains (warm-start),
  * HOLDS each perturbation for HOLD_LEN trials and updates on the settled response,
  * uses a diagonal, k-clamped estimator that cannot blow up to k = -2.
None of this changes the MATLAB loop -- the agent simply returns the same action on
the held trials. See LIVE_KWARGS below to tune or disable.

Place this file wherever MATLAB will add it to py.sys.path (AGENT_BRIDGE_DIR in the
.m file). It adds AGENT_CODE_DIR (your existing Python sim folder) to sys.path so
adaptive_agent.py is importable.

FUNCTIONAL REFACTOR
-------------------
The bridge now uses an immutable BridgeContext dataclass instead of a mutable
global _global_agent.  make_agent() replaces the entire context reference rather
than mutating a shared agent in place.
"""

import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from adaptive_agent import AdaptiveAgent, AdaptiveGEstimatorAgent, make_probe_schedule

# Folder that contains adaptive_agent.py, population_sim.py, skeleton_loop.py.
AGENT_CODE_DIR = r"C:\Users\mitti\Downloads"
if AGENT_CODE_DIR not in sys.path:
    sys.path.insert(0, AGENT_CODE_DIR)

_global_agent: Optional[AdaptiveGEstimatorAgent] = None

@dataclass(frozen=True)
class BridgeContext:
    """Immutable reference to the current agent held by the bridge."""
    agent: Optional[AdaptiveAgent] = None


# Module-level context — replaced (not mutated) by make_agent()
_context: Optional[BridgeContext] = None

# ---- live-session defaults (tune here) -----------------------------------------
LIVE_KWARGS: Dict[str, Any] = dict(
    diagonal_only=True,
    k_clamp=(0.05, 0.70),
    hold_len=4,
    settle_window=2,  # average the last 2 takes of each hold as "settled"
    probe_amp=80.0,
    probe_axes=2,  # [(+P,0),(0,+P)]; 4 adds the negative pair (more trials)
    explore0=0.0,
    lam=0.99,
    gain_prior=0.28,
)
# Trial-budget note (per session): act() is first called on the baseline anchor, then
# once per valid take. probe costs probe_axes*hold_len trials, each control block
# costs hold_len. With hold_len=4, probe_axes=2 -> 8 probe trials; the rest are
# control. Consider lowering ADAPTIVE_NUM_BASELINE in the .m to free trials.
# --------------------------------------------------------------------------------


def make_agent(
    target_f1: float, target_f2: float, seed: int = 0, **overrides: Any
) -> bool:
    """Create the live agent. Pass keyword overrides to change any LIVE_KWARGS
    (or set live=False for the original per-trial behavior)."""
    global _context
    agent_config = dict(LIVE_KWARGS)
    agent_config.update(overrides)

    if agent_config.pop("live", True):
        probe = make_probe_schedule(
            agent_config.pop("probe_amp", 80.0), agent_config.pop("probe_axes", 2)
        )
        agent = AdaptiveGEstimatorAgent(
            target=[float(target_f1), float(target_f2)],
            seed=int(seed),
            diagonal_only=agent_config.get("diagonal_only", True),
            k_clamp=agent_config.get("k_clamp", (0.05, 0.70)),
            hold_len=agent_config.get("hold_len", 4),
            settle_window=agent_config.get("settle_window", 2),
            probe_schedule=probe,
            explore0=agent_config.get("explore0", 0.0),
            lam=agent_config.get("lam", 0.99),
            gain_prior=agent_config.get("gain_prior", 0.28),
        )
    else:
        # original single-trial agent (no probe / hold / clamp)
        agent = AdaptiveGEstimatorAgent(
            target=[float(target_f1), float(target_f2)],
            gain_prior=agent_config.get("gain_prior", 0.28),
            lam=agent_config.get("lam", 0.99),
            explore0=agent_config.get("explore0", 20.0),
            seed=int(seed),
        )
    _context = BridgeContext(agent=agent)
    return True


def reset_session() -> bool:
    """Start a session. Keeps the learned estimate / probe map (warm start across
    sessions); clears only the within-session bookkeeping."""
    if _context is None or _context.agent is None:
        return True
    _context.agent.reset_block_state(keep_estimate=True)
    return True


def act(obs_f1: float, obs_f2: float) -> List[float]:
    """Feed the measured mid-vowel formants in; get the next perturbation back.

    Returns [dF1, dF2] in Hz as a plain Python list. During a hold the SAME action
    is returned for several trials by design -- apply whatever comes back."""
    if _context is None or _context.agent is None:
        raise RuntimeError("agent_bridge.make_agent(...) must be called first")
    action_result = _context.agent.act(
        np.array([float(obs_f1), float(obs_f2)], dtype=float)
    )
    return [float(action_result[0]), float(action_result[1])]


def skip_trial() -> bool:
    """Call on a trial where NO valid vowel was detected.

    Live mode: act() is only ever called with valid takes, so the held perturbation
    simply persists and the block waits for the next valid take -- nothing to undo.
    Simple mode: invalidate the pending one-step pairing (keeps the learned estimate)."""
    if _context is None or _context.agent is None:
        return True
    if _context.agent.live:
        return True  # holds tolerate a missing take with no action needed
    _context.agent.reset_block_state(keep_estimate=True)
    return True


def get_estimate() -> List[float]:
    """Return [k_F1, k_F2, cross_F1<-F2, cross_F2<-F1] from the current estimated_G_matrix.
    AdaptationRun.m logs this each trial; diagonal mode reports ~0 cross terms."""
    if _context is None or _context.agent is None:
        return [0.0, 0.0, 0.0, 0.0]
    G = _context.agent.estimated_G_matrix
    return [
        float(-G[0, 0]),
        float(-G[1, 1]),
        float(G[0, 1]),
        float(G[1, 0]),
    ]


def get_map() -> List[float]:
    """Return the per-speaker map identified by the probe, as a plain dict-like
    list [baseline_f1, baseline_f2, k_f1, k_f2], or zeros if the probe hasn't run.
    This is the hook for normalizing targets into per-speaker space."""
    if _context is None or _context.agent is None:
        return [0.0, 0.0, 0.0, 0.0]
    speaker_map = _context.agent.speaker_map
    if not speaker_map:
        return [0.0, 0.0, 0.0, 0.0]
    baseline = speaker_map["baseline"]
    gains = speaker_map["k"]
    return [float(baseline[0]), float(baseline[1]), float(gains[0]), float(gains[1])]


def get_phase() -> str:
    """'probe' or 'control' -- useful for on-screen status during a live run."""
    if _context is None or _context.agent is None:
        return "control"
    return _context.agent.phase
