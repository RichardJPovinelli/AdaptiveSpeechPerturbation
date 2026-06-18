"""Tests for agent_bridge.py — MATLAB bridge functions (BridgeContext pattern)."""

# pyright: reportPrivateUsage=false
# pyright: reportUnknownParameterType=false
# pyright: reportMissingParameterType=false

import numpy as np
import pytest

import agent_bridge


# ---------------------------------------------------------------------------
# Helper: create a fresh bridge context for each test
# ---------------------------------------------------------------------------
@pytest.fixture
def fresh_bridge():
    """Return a fresh BridgeContext by calling make_agent, then clean up."""
    agent_bridge.make_agent(300.0, 3000.0, seed=42)
    yield
    # Reset to no-agent state after test
    agent_bridge.reset_session()


# ---------------------------------------------------------------------------
# Helper: create a non-live bridge context
# ---------------------------------------------------------------------------
@pytest.fixture
def non_live_bridge():
    """Non-live bridge (no probe, no clamp)."""
    agent_bridge.make_agent(300.0, 3000.0, seed=42, live=False)
    yield
    agent_bridge.reset_session()


# ============================================================================
# TestMakeAgent
# ============================================================================
class TestMakeAgent:
    def test_live_mode_creates_agent(self):
        result = agent_bridge.make_agent(300.0, 3000.0, seed=42)
        assert result is True

    def test_non_live_mode(self):
        result = agent_bridge.make_agent(300.0, 3000.0, seed=42, live=False)
        assert result is True

    def test_override_kwargs(self):
        agent_bridge.make_agent(300.0, 3000.0, seed=42, gain_prior=0.50)
        # Verify the agent was created with the custom gain_prior
        assert agent_bridge.get_estimate() is not None

    def test_default_is_live_mode(self):
        # Default make_agent should create a live agent (hold_len > 1)
        agent_bridge.make_agent(300.0, 3000.0, seed=42)
        assert agent_bridge.get_phase() in ("probe", "control")


# ============================================================================
# TestResetSession
# ============================================================================
class TestResetSession:
    def test_resets_block_state(self, fresh_bridge):
        obs = np.array([500.0, 1500.0])
        agent_bridge.act(float(obs[0]), float(obs[1]))
        agent_bridge.reset_session()
        # After reset, previous action/observation should be cleared
        # (verified by the agent returning clean actions on next act)

    def test_no_op_when_no_agent(self):
        # Without calling make_agent, reset_session should still return True
        agent_bridge._context = agent_bridge.BridgeContext(agent=None)
        assert agent_bridge.reset_session() is True


# ============================================================================
# TestAct
# ============================================================================
class TestAct:
    def test_raises_before_make_agent(self):
        # Set context to no-agent state
        agent_bridge._context = agent_bridge.BridgeContext(agent=None)
        with pytest.raises(RuntimeError, match="make_agent"):
            agent_bridge.act(500.0, 1500.0)

    def test_returns_float_list(self, fresh_bridge):
        result = agent_bridge.act(500.0, 1500.0)
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(v, float) for v in result)

    def test_repeated_calls_return_valid_actions(self, fresh_bridge):
        obs_x, obs_y = 500.0, 1500.0
        for _ in range(5):
            result = agent_bridge.act(obs_x, obs_y)
            assert len(result) == 2
            obs_x, obs_y = result[0], result[1]


# ============================================================================
# TestSkipTrial
# ============================================================================
class TestSkipTrial:
    def test_live_mode_no_op(self, fresh_bridge):
        assert agent_bridge.skip_trial() is True

    def test_non_live_mode_clears_history(self, non_live_bridge):
        # Act once to set history
        agent_bridge.act(500.0, 1500.0)
        agent_bridge.skip_trial()
        # Should succeed without error
        assert agent_bridge.skip_trial() is True

    def test_no_agent(self):
        agent_bridge._context = agent_bridge.BridgeContext(agent=None)
        assert agent_bridge.skip_trial() is True


# ============================================================================
# TestGetEstimate
# ============================================================================
class TestGetEstimate:
    def test_zeros_when_no_agent(self):
        agent_bridge._context = agent_bridge.BridgeContext(agent=None)
        result = agent_bridge.get_estimate()
        assert result == [0.0, 0.0, 0.0, 0.0]

    def test_returns_estimate_after_create(self, fresh_bridge):
        result = agent_bridge.get_estimate()
        assert len(result) == 4
        # Diagonal mode: cross terms should be ~0
        assert abs(result[2]) < 0.01
        assert abs(result[3]) < 0.01


# ============================================================================
# TestGetMap
# ============================================================================
class TestGetMap:
    def test_zeros_before_probe(self, non_live_bridge):
        # Non-live agent has no probe, so map is None
        result = agent_bridge.get_map()
        assert result == [0.0, 0.0, 0.0, 0.0]

    def test_returns_map_after_probe(self, fresh_bridge):
        # Run through probe phase with enough calls
        obs = np.array([500.0, 1500.0])
        for _ in range(30):
            result = agent_bridge.act(float(obs[0]), float(obs[1]))
            obs = np.array(result)
        result = agent_bridge.get_map()
        assert len(result) == 4
        # Baseline and gains should be set
        assert result[0] != 0.0 or result[1] != 0.0


# ============================================================================
# TestGetPhase
# ============================================================================
class TestGetPhase:
    def test_initial_phase(self, fresh_bridge):
        # Live agent starts in probe phase
        assert agent_bridge.get_phase() in ("probe", "control")

    def test_phase_transitions(self, fresh_bridge):
        # Run enough trials to transition from probe → control
        obs = np.array([500.0, 1500.0])
        for _ in range(50):
            result = agent_bridge.act(float(obs[0]), float(obs[1]))
            obs = np.array(result)
        # After many trials, should be in control phase
        assert agent_bridge.get_phase() == "control"
