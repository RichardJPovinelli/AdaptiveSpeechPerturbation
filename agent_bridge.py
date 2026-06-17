"""
agent_bridge.py
===============
Thin bridge so MATLAB (AdaptationRun.m) can drive the Python AdaptiveGEstimatorAgent
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
"""

import sys

# Folder that contains adaptive_agent.py, population_sim.py, skeleton_loop.py.
AGENT_CODE_DIR = r"C:\Users\mitti\Downloads"
if AGENT_CODE_DIR not in sys.path:
    sys.path.insert(0, AGENT_CODE_DIR)

import numpy as np
from adaptive_agent import AdaptiveGEstimatorAgent, make_probe_schedule

_agent = None

# ---- live-session defaults (tune here) -----------------------------------------
# HOLD_LEN: trials each perturbation is held so slow compensation can build before we
#   read it. PROBE_AXES/PROBE_AMP: the opening per-speaker identification block.
#   K_CLAMP: (min,max) physiological gain; the positive floor keeps the inverse
#   controller stable (a near-zero k would otherwise demand an enormous action).
#   Set min negative (e.g. -0.20) if you explicitly want to model followers, but
#   expect noisier control near k=0.
LIVE_KWARGS = dict(
    diagonal_only=True,
    k_clamp=(0.05, 0.70),
    hold_len=4,
    settle_window=2,          # average the last 2 takes of each hold as "settled"
    probe_amp=80.0,
    probe_axes=2,             # [(+P,0),(0,+P)]; 4 adds the negative pair (more trials)
    explore0=0.0,
    lam=0.99,
    gain_prior=0.28,
)
# Trial-budget note (per session): act() is first called on the baseline anchor, then
# once per valid take. probe costs probe_axes*hold_len trials, each control block
# costs hold_len. With hold_len=4, probe_axes=2 -> 8 probe trials; the rest are
# control. Consider lowering ADAPTIVE_NUM_BASELINE in the .m to free trials.
# --------------------------------------------------------------------------------


def make_agent(target_f1, target_f2, seed=0, **overrides):
    """Create the live agent. Pass keyword overrides to change any LIVE_KWARGS
    (or set live=False for the original per-trial behavior)."""
    global _agent
    cfg = dict(LIVE_KWARGS)
    cfg.update(overrides)

    if cfg.pop("live", True):
        probe = make_probe_schedule(cfg.pop("probe_amp", 80.0),
                                    cfg.pop("probe_axes", 2))
        _agent = AdaptiveGEstimatorAgent(
            target=[float(target_f1), float(target_f2)],
            seed=int(seed),
            diagonal_only=cfg.get("diagonal_only", True),
            k_clamp=cfg.get("k_clamp", (0.05, 0.70)),
            hold_len=cfg.get("hold_len", 4),
            settle_window=cfg.get("settle_window", 2),
            probe_schedule=probe,
            explore0=cfg.get("explore0", 0.0),
            lam=cfg.get("lam", 0.99),
            gain_prior=cfg.get("gain_prior", 0.28),
        )
    else:
        # original single-trial agent (no probe / hold / clamp)
        _agent = AdaptiveGEstimatorAgent(
            target=[float(target_f1), float(target_f2)],
            gain_prior=cfg.get("gain_prior", 0.28),
            lam=cfg.get("lam", 0.99),
            explore0=cfg.get("explore0", 20.0),
            seed=int(seed),
        )
    return True


def reset_session():
    """Start a session. Keeps the learned estimate / probe map (warm start across
    sessions); clears only the within-session bookkeeping."""
    global _agent
    if _agent is None:
        return True
    if hasattr(_agent, "reset_block_state"):
        _agent.reset_block_state(keep_estimate=True)
    # also clear the simple-path history (harmless for live agents)
    _agent.prev_a = None
    _agent.prev_obs = None
    return True


def act(obs_f1, obs_f2):
    """Feed the measured mid-vowel formants in; get the next perturbation back.

    Returns [dF1, dF2] in Hz as a plain Python list. During a hold the SAME action
    is returned for several trials by design -- apply whatever comes back."""
    global _agent
    if _agent is None:
        raise RuntimeError("agent_bridge.make_agent(...) must be called first")
    a = _agent.act(np.array([float(obs_f1), float(obs_f2)], dtype=float))
    return [float(a[0]), float(a[1])]


def skip_trial():
    """Call on a trial where NO valid vowel was detected.

    Live mode: act() is only ever called with valid takes, so the held perturbation
    simply persists and the block waits for the next valid take -- nothing to undo.
    Simple mode: invalidate the pending one-step pairing (keeps the learned W)."""
    global _agent
    if _agent is None:
        return True
    if getattr(_agent, "_live", False):
        return True  # holds tolerate a missing take with no action needed
    _agent.prev_a = None
    _agent.prev_obs = None
    return True


def get_estimate():
    """Return [k_F1, k_F2, cross_F1<-F2, cross_F2<-F1] from the current W.
    AdaptationRun.m logs this each trial; diagonal mode reports ~0 cross terms."""
    global _agent
    if _agent is None:
        return [0.0, 0.0, 0.0, 0.0]
    W = _agent.W
    return [float(-W[0, 0]), float(-W[1, 1]), float(W[0, 1]), float(W[1, 0])]


def get_map():
    """Return the per-speaker map identified by the probe, as a plain dict-like
    list [baseline_f1, baseline_f2, k_f1, k_f2], or zeros if the probe hasn't run.
    This is the hook for normalizing targets into per-speaker space."""
    global _agent
    m = getattr(_agent, "speaker_map", None) if _agent is not None else None
    if not m:
        return [0.0, 0.0, 0.0, 0.0]
    b = m["baseline"]; k = m["k"]
    return [float(b[0]), float(b[1]), float(k[0]), float(k[1])]


def get_phase():
    """'probe' or 'control' -- useful for on-screen status during a live run."""
    global _agent
    return getattr(_agent, "phase", "control") if _agent is not None else "control"
