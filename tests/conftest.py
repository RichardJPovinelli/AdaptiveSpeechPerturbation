"""Shared pytest fixtures for adaptive_agent and agent_bridge tests."""

import numpy as np
import pytest

from adaptive_agent import AdaptiveGEstimatorAgent, make_probe_schedule


# ---------------------------------------------------------------------------
# Fixtures: simple (non-live) agent
# ---------------------------------------------------------------------------
@pytest.fixture
def simple_agent() -> AdaptiveGEstimatorAgent:
    """Agent with default (non-live) settings — no probe, no clamp, hold_len=1."""
    return AdaptiveGEstimatorAgent(
        target=np.array([300.0, 3000.0]),
        gain_prior=0.28,
        seed=42,
    )


# ---------------------------------------------------------------------------
# Fixtures: live agent (probe + clamp + hold)
# ---------------------------------------------------------------------------
@pytest.fixture
def live_agent() -> AdaptiveGEstimatorAgent:
    """Agent with live-session robustness enabled (probe, diagonal, k-clamp, hold)."""
    probe = make_probe_schedule(probe_amp=80.0, n_axes=2)
    return AdaptiveGEstimatorAgent(
        target=np.array([300.0, 3000.0]),
        seed=42,
        diagonal_only=True,
        k_clamp=(0.05, 0.70),
        hold_len=4,
        settle_window=2,
        probe_schedule=probe,
        explore0=0.0,
        lam=0.99,
        gain_prior=0.28,
    )


# ---------------------------------------------------------------------------
# Fixtures: synthetic observation values
# ---------------------------------------------------------------------------
@pytest.fixture
def baseline_obs() -> np.ndarray:
    """A synthetic baseline production [F1, F2]."""
    return np.array([500.0, 1500.0])


@pytest.fixture
def shifted_obs() -> np.ndarray:
    """A synthetic observation shifted from baseline (simulating compensation)."""
    return np.array([450.0, 1450.0])
