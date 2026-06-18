"""Shared pytest fixtures for adaptive_agent and agent_bridge tests."""

import numpy as np
import pytest

from adaptive_agent import (
    AgentConfig,
    AgentState,
    AdaptiveGEstimatorAgent,
    default_state,
    make_probe_schedule,
)


# ---------------------------------------------------------------------------
# Fixtures: AgentConfig (pure functional style)
# ---------------------------------------------------------------------------
@pytest.fixture
def simple_config() -> AgentConfig:
    """Minimal config — non-live, no probe, no clamp, hold_len=1."""
    return AgentConfig(
        target=np.array([300.0, 3000.0]),
        gain_prior=0.28,
        seed=42,
    )


@pytest.fixture
def live_config() -> AgentConfig:
    """Config with live-session robustness (probe, diagonal, k-clamp, hold)."""
    probe = make_probe_schedule(probe_amp=80.0, n_axes=2)
    return AgentConfig(
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
# Fixtures: AgentState (pure functional style)
# ---------------------------------------------------------------------------
@pytest.fixture
def default_simple_state(simple_config: AgentConfig) -> AgentState:
    """Fresh default state for the simple config."""
    return default_state(simple_config)


@pytest.fixture
def default_live_state(live_config: AgentConfig) -> AgentState:
    """Fresh default state for the live config."""
    return default_state(live_config)


# ---------------------------------------------------------------------------
# Fixtures: wrapper agents (backward-compatible API)
# ---------------------------------------------------------------------------
@pytest.fixture
def simple_agent() -> AdaptiveGEstimatorAgent:
    """Agent wrapper with default (non-live) settings."""
    return AdaptiveGEstimatorAgent(
        target=np.array([300.0, 3000.0]),
        gain_prior=0.28,
        seed=42,
    )


@pytest.fixture
def live_agent() -> AdaptiveGEstimatorAgent:
    """Agent wrapper with live-session robustness enabled."""
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
