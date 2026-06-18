"""Tests for agent_bridge.py — MATLAB bridge functions."""

import importlib
import numpy as np
import pytest

import agent_bridge
from adaptive_agent import make_probe_schedule


# Helper to reload the bridge module between tests to reset global state
@pytest.fixture(autouse=True)
def reset_bridge():
    """Reset _global_agent before each test to avoid state leakage."""
    importlib.reload(agent_bridge)
    yield
    # Ensure clean state after test
    agent_bridge._global_agent = None


# ============================================================================
# TestMakeAgent
# ============================================================================
class TestMakeAgent:
    def test_live_mode_creates_agent(self):
        result = agent_bridge.make_agent(300.0, 3000.0, seed=42)
        assert result is True
        assert agent_bridge._global_agent is not None
        assert agent_bridge._global_agent._live

    def test_non_live_mode(self):
        result = agent_bridge.make_agent(300.0, 3000.0, seed=42, live=False)
        assert result is True
        assert agent_bridge._global_agent is not None
        assert not agent_bridge._global_agent._live

    def test_override_kwargs(self):
        agent_bridge.make_agent(300.0, 3000.0, seed=42, gain_prior=0.50)
        assert agent_bridge._global_agent.gain_prior == 0.50


# ============================================================================
# TestResetSession
# ============================================================================
class TestResetSession:
    def test_resets_block_state(self):
        agent_bridge.make_agent(300.0, 3000.0, seed=42)
        obs = np.array([500.0, 1500.0])
        agent_bridge._global_agent._act_live(obs)
        agent_bridge.reset_session()
        assert agent_bridge._global_agent.previous_action is None
        assert agent_bridge._global_agent.previous_observation is None

    def test_no_op_when_no_agent(self):
        assert agent_bridge.reset_session() is True


# ============================================================================
# TestAct
# ============================================================================
class TestAct:
    def test_raises_before_make_agent(self):
        with pytest.raises(RuntimeError, match="make_agent"):
            agent_bridge.act(500.0, 1500.0)

    def test_returns_float_list(self):
        agent_bridge.make_agent(300.0, 3000.0, seed=42)
        result = agent_bridge.act(500.0, 1500.0)
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(v, float) for v in result)


# ============================================================================
# TestSkipTrial
# ============================================================================
class TestSkipTrial:
    def test_live_mode_no_op(self):
        agent_bridge.make_agent(300.0, 3000.0, seed=42)
        assert agent_bridge.skip_trial() is True

    def test_simple_mode_clears_history(self):
        agent_bridge.make_agent(300.0, 3000.0, seed=42, live=False)
        agent_bridge._global_agent.previous_action = np.array([1.0, 2.0])
        agent_bridge._global_agent.previous_observation = np.array([3.0, 4.0])
        agent_bridge.skip_trial()
        assert agent_bridge._global_agent.previous_action is None
        assert agent_bridge._global_agent.previous_observation is None

    def test_no_agent(self):
        assert agent_bridge.skip_trial() is True


# ============================================================================
# TestGetEstimate
# ============================================================================
class TestGetEstimate:
    def test_zeros_when_no_agent(self):
        result = agent_bridge.get_estimate()
        assert result == [0.0, 0.0, 0.0, 0.0]

    def test_returns_estimate_after_create(self):
        agent_bridge.make_agent(300.0, 3000.0, seed=42)
        result = agent_bridge.get_estimate()
        assert len(result) == 4
        # Diagonal mode: cross terms should be ~0
        assert abs(result[2]) < 0.01
        assert abs(result[3]) < 0.01


# ============================================================================
# TestGetMap
# ============================================================================
class TestGetMap:
    def test_zeros_before_probe(self):
        # Non-live agent has no probe, so map is None
        agent_bridge.make_agent(300.0, 3000.0, seed=42, live=False)
        result = agent_bridge.get_map()
        assert result == [0.0, 0.0, 0.0, 0.0]

    def test_returns_map_after_probe(self):
        agent_bridge.make_agent(300.0, 3000.0, seed=42)
        # Run through probe phase
        obs = np.array([500.0, 1500.0])
        calls_per_probe = agent_bridge._global_agent.hold_len + 1
        total_calls = calls_per_probe * len(agent_bridge._global_agent.probe_schedule)
        for _ in range(total_calls):
            obs = np.array(agent_bridge.act(float(obs[0]), float(obs[1])))
        result = agent_bridge.get_map()
        assert len(result) == 4
        # Baseline and gains should be set
        assert result[0] != 0.0 or result[1] != 0.0


# ============================================================================
# TestGetPhase
# ============================================================================
class TestGetPhase:
    def test_probe_phase(self):
        agent_bridge.make_agent(300.0, 3000.0, seed=42)
        assert agent_bridge.get_phase() == "probe"

    def test_control_phase_after_probe(self):
        agent_bridge.make_agent(300.0, 3000.0, seed=42)
        # Exhaust probe
        obs = np.array([500.0, 1500.0])
        calls_per_probe = agent_bridge._global_agent.hold_len + 1
        total_calls = calls_per_probe * len(agent_bridge._global_agent.probe_schedule)
        for _ in range(total_calls):
            obs = np.array(agent_bridge.act(float(obs[0]), float(obs[1])))
        assert agent_bridge.get_phase() == "control"
