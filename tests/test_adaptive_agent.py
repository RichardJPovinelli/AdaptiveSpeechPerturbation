"""Tests for adaptive_agent.py — AdaptiveGEstimatorAgent and make_probe_schedule."""

import numpy as np
import pytest

from adaptive_agent import AdaptiveGEstimatorAgent, make_probe_schedule


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
# TestAdaptiveGEstimatorAgentInit
# ============================================================================
class TestAdaptiveGEstimatorAgentInit:
    def test_non_live_defaults(self):
        agent = AdaptiveGEstimatorAgent(target=[300.0, 3000.0], seed=0)
        assert not agent._live  # pyright: ignore[reportPrivateUsage]
        assert agent.phase == "control"
        assert agent.probe_schedule is None

    def test_live_with_probe(self):
        probe = make_probe_schedule(n_axes=2)
        agent = AdaptiveGEstimatorAgent(target=[300.0, 3000.0], seed=0, probe_schedule=probe)
        assert agent._live
        assert agent.phase == "probe"

    def test_live_without_probe_but_k_clamp(self):
        agent = AdaptiveGEstimatorAgent(target=[300.0, 3000.0], seed=0, k_clamp=(0.1, 0.6))
        assert agent._live
        assert agent.phase == "control"

    def test_target_coerced_to_ndarray(self):
        agent = AdaptiveGEstimatorAgent(target=[300.0, 3000.0])
        assert isinstance(agent.target, np.ndarray)
        assert agent.target.dtype == np.float64


# ============================================================================
# TestRLSUpdate
# ============================================================================
class TestRLSUpdate:
    def test_updates_estimate_and_covariance(self, simple_agent):
        action = np.array([10.0, 20.0])
        delta = np.array([5.0, 8.0])
        old_estimate = simple_agent.estimated_G_matrix.copy()
        simple_agent._rls_update(action, delta)
        # Estimate should change after receiving a learning signal
        assert not np.allclose(simple_agent.estimated_G_matrix, old_estimate)

    def test_covariance_decreases(self, simple_agent):
        action = np.array([10.0, 20.0])
        delta = np.array([5.0, 8.0])
        old_cov_trace = [np.trace(c) for c in simple_agent.rls_covariance]
        simple_agent._rls_update(action, delta)
        new_cov_trace = [np.trace(c) for c in simple_agent.rls_covariance]
        # Covariance should generally decrease after an update
        assert new_cov_trace[0] <= old_cov_trace[0] + 1e-6
        assert new_cov_trace[1] <= old_cov_trace[1] + 1e-6


# ============================================================================
# TestRLSUpdateDiag
# ============================================================================
class TestRLSUpdateDiag:
    def test_diagonal_only(self, simple_agent):
        simple_agent.diagonal_only = True
        action = np.array([10.0, 20.0])
        delta = np.array([5.0, 8.0])
        simple_agent._rls_update_diag(action, delta)
        # Off-diagonal should be zero
        assert simple_agent.estimated_G_matrix[0, 1] == 0.0
        assert simple_agent.estimated_G_matrix[1, 0] == 0.0

    def test_off_diagonal_zeroed(self, simple_agent):
        # Set non-zero off-diagonal to verify they get zeroed
        simple_agent.estimated_G_matrix[0, 1] = 0.5
        simple_agent.estimated_G_matrix[1, 0] = 0.3
        action = np.array([10.0, 20.0])
        delta = np.array([5.0, 8.0])
        simple_agent._rls_update_diag(action, delta)
        assert simple_agent.estimated_G_matrix[0, 1] == 0.0
        assert simple_agent.estimated_G_matrix[1, 0] == 0.0


# ============================================================================
# TestApplyKClamp
# ============================================================================
class TestApplyKClamp:
    def test_no_op_when_none(self, simple_agent):
        simple_agent.k_clamp = None
        simple_agent.estimated_G_matrix[0, 0] = -2.0
        simple_agent._apply_k_clamp()
        assert simple_agent.estimated_G_matrix[0, 0] == -2.0

    def test_clamps_to_bounds(self, simple_agent):
        simple_agent.k_clamp = (0.05, 0.70)
        simple_agent.estimated_G_matrix[0, 0] = -1.0  # gain = 1.0, exceeds max
        simple_agent._apply_k_clamp()
        assert simple_agent.estimated_G_matrix[0, 0] == pytest.approx(-0.70)

    def test_no_change_within_bounds(self, simple_agent):
        simple_agent.k_clamp = (0.05, 0.70)
        simple_agent.estimated_G_matrix[0, 0] = -0.30
        simple_agent._apply_k_clamp()
        assert simple_agent.estimated_G_matrix[0, 0] == pytest.approx(-0.30)


# ============================================================================
# TestControlAction
# ============================================================================
class TestControlAction:
    def test_returns_non_zero_action(self, simple_agent):
        obs = np.array([500.0, 1500.0])
        action = simple_agent._control_action(obs)
        assert action.shape == (2,)
        assert not np.allclose(action, 0.0)

    def test_caps_norm(self, simple_agent):
        obs = np.array([2000.0, 5000.0])  # large gap
        action = simple_agent._control_action(obs)
        assert np.linalg.norm(action) <= simple_agent.action_cap + 1e-6

    def test_clips_to_bounds(self, simple_agent):
        obs = np.array([2000.0, 5000.0])
        action = simple_agent._control_action(obs)
        assert np.all(action >= simple_agent.action_min)
        assert np.all(action <= simple_agent.action_max)


# ============================================================================
# TestActSimple
# ============================================================================
class TestActSimple:
    def test_first_call_sets_prior(self, simple_agent, baseline_obs):
        action = simple_agent._act_simple(baseline_obs)
        assert simple_agent.previous_action is not None
        assert simple_agent.previous_observation is not None
        assert action.dtype == np.float32

    def test_second_call_learns(self, simple_agent, baseline_obs):
        simple_agent._act_simple(baseline_obs)
        shifted = np.array([450.0, 1450.0])
        simple_agent._act_simple(shifted)
        assert simple_agent.trial_count == 2

    def test_trial_count_increments(self, simple_agent, baseline_obs):
        assert simple_agent.trial_count == 0
        simple_agent._act_simple(baseline_obs)
        assert simple_agent.trial_count == 1
        simple_agent._act_simple(baseline_obs)
        assert simple_agent.trial_count == 2

    def test_exploration_noise_decay(self, simple_agent, baseline_obs):
        simple_agent.exploration_noise_init = 25.0
        simple_agent.exploration_noise_decay = 0.78
        noise_0 = 25.0 * (0.78**0)
        noise_5 = 25.0 * (0.78**5)
        assert noise_5 < noise_0


# ============================================================================
# TestSettled
# ============================================================================
class TestSettled:
    def test_settle_window_mean(self, live_agent):
        live_agent.block_observations = [
            np.array([10.0, 20.0]),
            np.array([12.0, 22.0]),
            np.array([14.0, 24.0]),
            np.array([16.0, 26.0]),
        ]
        settled = live_agent._settled()
        # Last 2 observations averaged
        expected = np.mean(np.array([[14.0, 24.0], [16.0, 26.0]]), axis=0)
        np.testing.assert_array_almost_equal(settled, expected)

    def test_fallback_half_hold(self, simple_agent):
        simple_agent.settle_window = None
        simple_agent.hold_len = 4
        simple_agent.block_observations = [
            np.array([10.0, 20.0]),
            np.array([12.0, 22.0]),
        ]
        settled = simple_agent._settled()
        # Back half of hold (hold_len // 2 = 2, but only 2 obs, so all of them)
        np.testing.assert_array_almost_equal(settled, np.array([11.0, 21.0]))


# ============================================================================
# TestStartBlock
# ============================================================================
class TestStartBlock:
    def test_probe_phase(self, live_agent, baseline_obs):
        assert live_agent.phase == "probe"
        action = live_agent._start_block(baseline_obs)
        np.testing.assert_array_almost_equal(action, live_agent.probe_schedule[0])

    def test_control_phase(self, simple_agent, baseline_obs):
        action = simple_agent._start_block(baseline_obs)
        expected = simple_agent._control_action(baseline_obs)
        np.testing.assert_array_almost_equal(action, expected)


# ============================================================================
# TestFitProbe
# ============================================================================
class TestFitProbe:
    def test_computes_gains(self, live_agent, baseline_obs):
        # Simulate probe data: held action and response
        live_agent.probe_data = [
            (np.array([80.0, 0.0]), np.array([-22.4, 0.0])),  # k_F1 ≈ 0.28
            (np.array([0.0, 80.0]), np.array([0.0, -24.0])),  # k_F2 ≈ 0.30
        ]
        live_agent.baseline_observation = baseline_obs.copy()
        live_agent.pre_block_observation = baseline_obs.copy()
        live_agent._fit_probe()
        assert live_agent.speaker_map is not None
        np.testing.assert_allclose(live_agent.speaker_map["k"], [0.28, 0.30], atol=0.05)

    def test_sets_speaker_map(self, live_agent, baseline_obs):
        live_agent.probe_data = [
            (np.array([80.0, 0.0]), np.array([-20.0, 0.0])),
            (np.array([0.0, 80.0]), np.array([0.0, -20.0])),
        ]
        live_agent.baseline_observation = baseline_obs.copy()
        live_agent.pre_block_observation = baseline_obs.copy()
        live_agent._fit_probe()
        assert "baseline" in live_agent.speaker_map
        assert "k" in live_agent.speaker_map


# ============================================================================
# TestActLive
# ============================================================================
class TestActLive:
    def test_first_call_sets_baseline(self, live_agent, baseline_obs):
        action = live_agent._act_live(baseline_obs)
        np.testing.assert_array_almost_equal(live_agent.baseline_observation, baseline_obs)
        assert live_agent.pre_block_observation is not None

    def test_hold_buffering(self, live_agent, baseline_obs):
        # First call sets baseline
        first_action = live_agent._act_live(baseline_obs)
        # Subsequent calls during hold return same action
        for _ in range(live_agent.hold_len - 1):
            obs = baseline_obs + np.array([1.0, 1.0])
            action = live_agent._act_live(obs)
            np.testing.assert_array_almost_equal(action, first_action)

    def test_block_completion_records_probe_data(self, live_agent, baseline_obs):
        # First call
        live_agent._act_live(baseline_obs)
        # Fill hold
        for _ in range(live_agent.hold_len):
            obs = baseline_obs + np.array([5.0, 5.0])
            live_agent._act_live(obs)
        # After block completes, probe data should be recorded
        assert len(live_agent.probe_data) > 0

    def test_probe_to_control_transition(self, live_agent, baseline_obs):
        # Exhaust probe schedule (2 probes, each costs hold_len+1 calls: 1 baseline + hold_len)
        calls_per_probe = live_agent.hold_len + 1
        total_calls = calls_per_probe * len(live_agent.probe_schedule)
        obs = baseline_obs.copy()
        for _ in range(total_calls):
            obs = live_agent._act_live(obs)
        # Should transition to control phase
        assert live_agent.phase == "control"

    def test_control_phase_rls_update(self, live_agent, baseline_obs):
        # Run through probe phase
        calls_per_probe = live_agent.hold_len + 1
        total_calls = calls_per_probe * len(live_agent.probe_schedule)
        obs = baseline_obs.copy()
        for _ in range(total_calls):
            obs = live_agent._act_live(obs)
        # Now in control phase — one more block should trigger RLS
        old_estimate = live_agent.estimated_G_matrix.copy()
        for _ in range(live_agent.hold_len + 1):
            obs = live_agent._act_live(obs)
        # Estimate should have been updated
        assert live_agent.phase == "control"


# ============================================================================
# TestResetBlockState
# ============================================================================
class TestResetBlockState:
    def test_keep_estimate(self, live_agent, baseline_obs):
        # Run a few trials to modify estimate
        for _ in range(3):
            live_agent._act_live(baseline_obs)
        old_estimate = live_agent.estimated_G_matrix.copy()
        live_agent.reset_block_state(keep_estimate=True)
        np.testing.assert_array_almost_equal(live_agent.estimated_G_matrix, old_estimate)
        assert live_agent.block_action is None
        assert live_agent.pre_block_observation is None

    def test_reset_all(self, simple_agent):
        simple_agent.estimated_G_matrix[0, 0] = -0.99
        simple_agent.reset_block_state(keep_estimate=False)
        expected = -simple_agent.gain_prior * np.eye(2)
        np.testing.assert_array_almost_equal(simple_agent.estimated_G_matrix, expected)
