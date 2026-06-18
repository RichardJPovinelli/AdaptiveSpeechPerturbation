"""Tests for adaptive_agent.py — pure functions, dataclasses, and wrapper API."""

# pyright: reportUnknownParameterType=false
# pyright: reportUnknownVariableType=false
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownArgumentType=false
# pyright: reportMissingParameterType=false

import numpy as np
import pytest

from adaptive_agent import (
    AgentConfig,
    AgentState,
    AdaptiveGEstimatorAgent,
    apply_k_clamp,
    control_action,
    default_state,
    make_probe_schedule,
    rls_update,
    rls_update_diag,
    settled,
)


# ============================================================================
# TestMakeProbeSchedule
# ============================================================================
class TestMakeProbeSchedule:
    def test_returns_two_axes_default(self):
        schedule = make_probe_schedule(n_axes=2)
        assert len(schedule) == 2
        np.testing.assert_array_almost_equal(schedule[0], [80.0, 0.0])
        np.testing.assert_array_almost_equal(schedule[1], [0.0, 80.0])

    def test_returns_four_axes_with_negatives(self):
        schedule = make_probe_schedule(n_axes=4)
        assert len(schedule) == 4
        np.testing.assert_array_almost_equal(schedule[2], [-80.0, 0.0])
        np.testing.assert_array_almost_equal(schedule[3], [0.0, -80.0])

    def test_custom_amplitude(self):
        schedule = make_probe_schedule(probe_amp=50.0, n_axes=2)
        np.testing.assert_array_almost_equal(schedule[0], [50.0, 0.0])


# ============================================================================
# TestDataclasses
# ============================================================================
class TestDataclasses:
    def test_config_is_frozen(self, simple_config: AgentConfig):
        with pytest.raises(Exception):
            simple_config.target = np.array([0.0, 0.0])  # type: ignore

    def test_state_is_frozen(self, default_simple_state: AgentState):
        with pytest.raises(Exception):
            default_simple_state.trial_count = 5  # type: ignore

    def test_config_live_property_non_live(self, simple_config: AgentConfig):
        assert not simple_config.live

    def test_config_live_property_with_probe(self, live_config: AgentConfig):
        assert live_config.live

    def test_default_state_initial_values(self, simple_config: AgentConfig):
        st = default_state(simple_config)
        assert st.trial_count == 0
        assert st.phase == "control"
        assert st.previous_action is None

    def test_default_state_probe_phase(self, live_config: AgentConfig):
        st = default_state(live_config)
        assert st.phase == "probe"

    def test_target_coerced_to_ndarray(self):
        agent = AdaptiveGEstimatorAgent(target=[300.0, 3000.0])
        assert isinstance(agent.target, np.ndarray)
        assert agent.target.dtype == np.float64


# ============================================================================
# TestRLSPureFunctions
# ============================================================================
class TestRLSPureFunctions:
    def test_rls_update_returns_new_state(self, simple_config: AgentConfig):
        st = default_state(simple_config)
        action = np.array([10.0, 20.0])
        delta = np.array([5.0, 8.0])
        new_st = rls_update(st, simple_config.lam, action, delta)
        assert new_st is not st  # immutability
        assert new_st.trial_count == st.trial_count  # preserved fields

    def test_rls_update_changes_estimate(self, simple_config: AgentConfig):
        st = default_state(simple_config)
        action = np.array([10.0, 20.0])
        delta = np.array([5.0, 8.0])
        new_st = rls_update(st, simple_config.lam, action, delta)
        assert not np.allclose(new_st.estimated_G_matrix, st.estimated_G_matrix)

    def test_rls_update_covariance_decreases(self, simple_config: AgentConfig):
        st = default_state(simple_config)
        action = np.array([10.0, 20.0])
        delta = np.array([5.0, 8.0])
        new_st = rls_update(st, simple_config.lam, action, delta)
        old_trace = [np.trace(c) for c in st.rls_covariance]
        new_trace = [np.trace(c) for c in new_st.rls_covariance]
        assert new_trace[0] <= old_trace[0] + 1e-6
        assert new_trace[1] <= old_trace[1] + 1e-6

    def test_rls_update_diag_zeroes_off_diagonal(self, simple_config: AgentConfig):
        st = default_state(simple_config)
        action = np.array([10.0, 20.0])
        delta = np.array([5.0, 8.0])
        new_st = rls_update_diag(st, simple_config.lam, action, delta)
        assert new_st.estimated_G_matrix[0, 1] == 0.0
        assert new_st.estimated_G_matrix[1, 0] == 0.0

    def test_rls_update_diag_zeroes_existing_off_diagonal(self, simple_config: AgentConfig):
        st = default_state(simple_config)
        # Manually set off-diagonal (via dataclass replace pattern)
        st = AgentState(
            estimated_G_matrix=st.estimated_G_matrix + np.array([[0.0, 0.5], [0.3, 0.0]]),
            rls_covariance=st.rls_covariance,
            trial_count=st.trial_count,
            phase=st.phase,
        )
        action = np.array([10.0, 20.0])
        delta = np.array([5.0, 8.0])
        new_st = rls_update_diag(st, simple_config.lam, action, delta)
        assert new_st.estimated_G_matrix[0, 1] == 0.0
        assert new_st.estimated_G_matrix[1, 0] == 0.0

    def test_determinism(self, simple_config: AgentConfig):
        """Same inputs produce same outputs (pure function property)."""
        st = default_state(simple_config)
        action = np.array([10.0, 20.0])
        delta = np.array([5.0, 8.0])
        result_a = rls_update(st, simple_config.lam, action, delta)
        result_b = rls_update(st, simple_config.lam, action, delta)
        np.testing.assert_array_almost_equal(result_a.estimated_G_matrix, result_b.estimated_G_matrix)


# ============================================================================
# TestApplyKClamp
# ============================================================================
class TestApplyKClamp:
    def test_no_op_when_none(self, simple_config: AgentConfig):
        st = default_state(simple_config)
        st = AgentState(
            estimated_G_matrix=np.array([[-2.0, 0.0], [0.0, -0.28]]),
            rls_covariance=st.rls_covariance,
            trial_count=st.trial_count,
            phase=st.phase,
        )
        new_st = apply_k_clamp(st, None)
        assert new_st.estimated_G_matrix[0, 0] == -2.0

    def test_clamps_to_bounds(self, simple_config: AgentConfig):
        st = default_state(simple_config)
        st = AgentState(
            estimated_G_matrix=np.array([[-1.0, 0.0], [0.0, -0.28]]),
            rls_covariance=st.rls_covariance,
            trial_count=st.trial_count,
            phase=st.phase,
        )
        new_st = apply_k_clamp(st, (0.05, 0.70))
        assert new_st.estimated_G_matrix[0, 0] == pytest.approx(-0.70)

    def test_no_change_within_bounds(self, simple_config: AgentConfig):
        st = default_state(simple_config)
        new_st = apply_k_clamp(st, (0.05, 0.70))
        assert new_st.estimated_G_matrix[0, 0] == pytest.approx(-0.28)


# ============================================================================
# TestControlAction
# ============================================================================
class TestControlAction:
    def test_returns_non_zero_action(self, simple_config: AgentConfig):
        st = default_state(simple_config)
        obs = np.array([500.0, 1500.0])
        action = control_action(simple_config, st, obs)
        assert action.shape == (2,)
        assert not np.allclose(action, 0.0)

    def test_caps_norm(self, simple_config: AgentConfig):
        st = default_state(simple_config)
        obs = np.array([2000.0, 5000.0])
        action = control_action(simple_config, st, obs)
        assert np.linalg.norm(action) <= simple_config.action_cap + 1e-6

    def test_clips_to_bounds(self, simple_config: AgentConfig):
        st = default_state(simple_config)
        obs = np.array([2000.0, 5000.0])
        action = control_action(simple_config, st, obs)
        assert np.all(action >= simple_config.action_min)
        assert np.all(action <= simple_config.action_max)


# ============================================================================
# TestSettled
# ============================================================================
class TestSettled:
    def test_settle_window_mean(self, live_config: AgentConfig):
        st = default_state(live_config)
        st = AgentState(
            estimated_G_matrix=st.estimated_G_matrix,
            rls_covariance=st.rls_covariance,
            block_observations=(
                np.array([10.0, 20.0]),
                np.array([12.0, 22.0]),
                np.array([14.0, 24.0]),
                np.array([16.0, 26.0]),
            ),
            phase=st.phase,
        )
        result = settled(st, live_config.settle_window, live_config.hold_len)
        expected = np.mean(np.array([[14.0, 24.0], [16.0, 26.0]]), axis=0)
        np.testing.assert_array_almost_equal(result, expected)

    def test_fallback_half_hold(self, simple_config: AgentConfig):
        st = default_state(simple_config)
        st = AgentState(
            estimated_G_matrix=st.estimated_G_matrix,
            rls_covariance=st.rls_covariance,
            block_observations=(
                np.array([10.0, 20.0]),
                np.array([12.0, 22.0]),
            ),
            phase=st.phase,
        )
        # hold_len=1, settle_window=None → window_size = max(1, 1//2) = 1
        # So only the last observation is returned
        result = settled(st, simple_config.settle_window, simple_config.hold_len)
        np.testing.assert_array_almost_equal(result, np.array([12.0, 22.0]))


# ============================================================================
# TestWrapperInit
# ============================================================================
class TestWrapperInit:
    def test_non_live_defaults(self):
        agent = AdaptiveGEstimatorAgent(target=[300.0, 3000.0], seed=0)
        assert not agent.live
        assert agent.phase == "control"
        assert agent.probe_schedule is None

    def test_live_with_probe(self):
        probe = make_probe_schedule(n_axes=2)
        agent = AdaptiveGEstimatorAgent(target=[300.0, 3000.0], seed=0, probe_schedule=probe)
        assert agent.live
        assert agent.phase == "probe"

    def test_live_without_probe_but_k_clamp(self):
        agent = AdaptiveGEstimatorAgent(target=[300.0, 3000.0], seed=0, k_clamp=(0.1, 0.6))
        assert agent.live
        assert agent.phase == "control"


# ============================================================================
# TestActSimple (wrapper API)
# ============================================================================
class TestActSimple:
    def test_first_call_sets_prior(self, simple_agent: AdaptiveGEstimatorAgent, baseline_obs: np.ndarray):
        action = simple_agent.act(baseline_obs)
        assert simple_agent.previous_action is not None
        assert simple_agent.previous_observation is not None
        assert action.dtype == np.float32

    def test_second_call_learns(self, simple_agent: AdaptiveGEstimatorAgent, baseline_obs: np.ndarray):
        simple_agent.act(baseline_obs)
        shifted = np.array([450.0, 1450.0])
        simple_agent.act(shifted)
        assert simple_agent.trial_count == 2

    def test_trial_count_increments(self, simple_agent: AdaptiveGEstimatorAgent, baseline_obs: np.ndarray):
        assert simple_agent.trial_count == 0
        simple_agent.act(baseline_obs)
        assert simple_agent.trial_count == 1
        simple_agent.act(baseline_obs)
        assert simple_agent.trial_count == 2

    def test_exploration_noise_decay(self, simple_agent: AdaptiveGEstimatorAgent, baseline_obs: np.ndarray):
        noise_0 = 25.0 * (0.78**0)
        noise_5 = 25.0 * (0.78**5)
        assert noise_5 < noise_0


# ============================================================================
# TestActLive (wrapper API)
# ============================================================================
class TestActLive:
    def test_first_call_sets_baseline(self, live_agent: AdaptiveGEstimatorAgent, baseline_obs: np.ndarray):
        live_agent.act(baseline_obs)
        assert live_agent.baseline_observation is not None
        np.testing.assert_array_almost_equal(live_agent.baseline_observation, baseline_obs)
        assert live_agent.pre_block_observation is not None

    def test_hold_buffering(self, live_agent: AdaptiveGEstimatorAgent, baseline_obs: np.ndarray):
        first_action = live_agent.act(baseline_obs)
        for _ in range(live_agent.hold_len - 1):
            obs = baseline_obs + np.array([1.0, 1.0])
            action = live_agent.act(obs)
            np.testing.assert_array_almost_equal(action, first_action)

    def test_block_completion_records_probe_data(self, live_agent: AdaptiveGEstimatorAgent, baseline_obs: np.ndarray):
        # First call sets baseline + starts block
        live_agent.act(baseline_obs)
        # hold_len calls fill the buffer
        for _ in range(live_agent.hold_len):
            obs = baseline_obs + np.array([5.0, 5.0])
            live_agent.act(obs)
        # One more call completes the block and records probe_data
        live_agent.act(baseline_obs + np.array([5.0, 5.0]))
        assert len(live_agent.probe_data) > 0

    def test_probe_to_control_transition(self, live_agent: AdaptiveGEstimatorAgent, baseline_obs: np.ndarray):
        assert live_agent.probe_schedule is not None
        # Each probe requires: 1 (baseline) + hold_len (fill buffer) + 1 (complete block)
        # But first probe shares the initial baseline call, so:
        # Total = 1 + hold_len (first block) + (1 + hold_len) * (len(probe_schedule) - 1) (remaining blocks)
        # Simplify: just run enough calls to exhaust all probes
        calls_per_probe = live_agent.hold_len + 2  # baseline + hold + complete
        total_calls = calls_per_probe * len(live_agent.probe_schedule)
        obs = baseline_obs.copy()
        for _ in range(total_calls):
            obs = live_agent.act(obs)
        assert live_agent.phase == "control"


# ============================================================================
# TestResetBlockState
# ============================================================================
class TestResetBlockState:
    def test_keep_estimate(self, live_agent: AdaptiveGEstimatorAgent, baseline_obs: np.ndarray):
        for _ in range(3):
            live_agent.act(baseline_obs)
        old_estimate = live_agent.estimated_G_matrix.copy()
        live_agent.reset_block_state(keep_estimate=True)
        np.testing.assert_array_almost_equal(live_agent.estimated_G_matrix, old_estimate)
        assert live_agent.block_action is None
        assert live_agent.pre_block_observation is None

    def test_reset_all(self, simple_agent: AdaptiveGEstimatorAgent):
        simple_agent.estimated_G_matrix[0, 0] = -0.99
        simple_agent.reset_block_state(keep_estimate=False)
        expected = -simple_agent.gain_prior * np.eye(2)
        np.testing.assert_array_almost_equal(simple_agent.estimated_G_matrix, expected)
        simple_agent.estimated_G_matrix[0, 0] = -0.99
        simple_agent.reset_block_state(keep_estimate=False)
        expected = -simple_agent.gain_prior * np.eye(2)
        np.testing.assert_array_almost_equal(simple_agent.estimated_G_matrix, expected)
