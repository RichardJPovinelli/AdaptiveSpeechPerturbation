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

import numpy as np

BEST_FIXED_GAIN = 1.0


def make_probe_schedule(probe_amp=80.0, n_axes=2):
    """Axis-aligned probe perturbations for cold per-speaker identification.

    Each probe vector is held by the agent for `hold_len` trials; the settled
    compensation across the hold identifies that axis's gain. The default 2-axis
    schedule [(P,0),(0,P)] is the minimum to read k_F1 and k_F2 separately.
    Add +/- pairs (drift cancellation) or extra amplitudes if your trial budget
    allows -- each extra probe costs hold_len trials.
    """
    P = float(probe_amp)
    sched = [np.array([P, 0.0]), np.array([0.0, P]),
             np.array([-P, 0.0]), np.array([0.0, -P])]
    return sched[:max(1, int(n_axes))]


class AdaptiveGEstimatorAgent:
    # initializing agent
    def __init__(self, target, gain_prior=0.28, action_cap=120.0,
                 lam=0.995, ridge=1e-3, explore0=25.0, explore_decay=0.78,
                 action_low=-200., action_high=200., seed=0,
                 # --- live-session robustness; defaults OFF => identical sim behavior ---
                 diagonal_only=False, k_clamp=None,
                 hold_len=1, settle_window=None, probe_schedule=None):
        self.target = np.asarray(target, dtype=np.float64)
        # Prior G = population-mean diagonal (avg of MacDonald 2010's k_F1, k_F2)
        self.W = -gain_prior * np.eye(2)  # agent estimate of G
        self.P = [np.eye(2) * 0.5, np.eye(2) * 0.5]  # RLS covariance
        self.lam, self.ridge = lam, ridge  # RLS forgetting factor (long bc lam close to 1)
        self.action_cap = action_cap  # Max norm for control action
        self.explore0, self.explore_decay = explore0, explore_decay  # exploration noise
        self.lo, self.hi = action_low, action_high
        self.rng = np.random.default_rng(seed)
        self.prev_a = self.prev_obs = None  # history, meaningful when multiple sessions
        self.t = 0
        self.gain_prior = float(gain_prior)

        # --- live-session features ---
        self.diagonal_only = bool(diagonal_only)
        self.k_clamp = tuple(k_clamp) if k_clamp is not None else None
        self.hold_len = max(1, int(hold_len))
        self.settle_window = settle_window  # None -> back half of the hold
        self.probe_schedule = ([np.asarray(p, float) for p in probe_schedule]
                               if probe_schedule else None)
        # live mode engages if any robustness feature is requested
        self._live = (self.diagonal_only or self.k_clamp is not None
                      or self.hold_len > 1 or self.probe_schedule is not None)

        # live state machine
        self.phase = "probe" if self.probe_schedule else "control"
        self.probe_idx = 0
        self.probe_data = []        # list of (held_action, settled_response)
        self.block_action = None
        self.block_obs = []
        self.pre_block_obs = None   # settled production at the start of this block
        self.anchor0 = None         # the very first (baseline) observation
        self.speaker_map = None     # {"baseline": [F1,F2], "k": [kF1,kF2]} after probe

    # ===================== estimator updates =====================
    def _rls_update(self, a, dy):
        # ORIGINAL full-matrix RLS (unchanged) -- the simulation path uses this.
        for i in range(2):
            P = self.P[i]
            Pa = P @ a
            # update filter coefficients
            K = Pa / (self.lam + a @ Pa)
            self.W[i] = self.W[i] + K * (dy[i] - self.W[i] @ a)
            self.P[i] = (P - np.outer(K, Pa)) / self.lam

    def _rls_update_diag(self, a, dy):
        # Two independent scalar RLS filters: dy[i] ~ W[i,i] * a[i]. No cross terms,
        # so the off-diagonal can never absorb noise and blow up.
        for i in range(2):
            ai = float(a[i])
            Pii = float(self.P[i][i, i])
            denom = self.lam + ai * Pii * ai
            K = (Pii * ai) / denom if denom != 0.0 else 0.0
            self.W[i, i] = self.W[i, i] + K * (dy[i] - self.W[i, i] * ai)
            self.P[i][i, i] = (Pii - K * ai * Pii) / self.lam
        self.W[0, 1] = self.W[1, 0] = 0.0

    def _apply_k_clamp(self):
        if self.k_clamp is None:
            return
        kmin, kmax = self.k_clamp
        for i in range(2):
            k = -self.W[i, i]
            self.W[i, i] = -float(np.clip(k, kmin, kmax))

    # ===================== control law =====================
    def _control_action(self, obs):
        d = self.target - np.asarray(obs, dtype=np.float64)  # gap to target
        # Solve, then perturb opposite (W ~ -k*I  =>  a ~ -d/k)
        a = np.linalg.solve(self.W.T @ self.W + self.ridge * np.eye(2),
                            self.W.T @ d)
        n = np.linalg.norm(a)
        if n > self.action_cap:
            a *= self.action_cap / n
        return np.clip(a, self.lo, self.hi)

    # ===================== public act =====================
    def act(self, observation):
        obs = np.asarray(observation, dtype=np.float64)
        if not self._live:
            return self._act_simple(obs)
        return self._act_live(obs)

    def _act_simple(self, obs):
        # EXACT original behavior (simulation path) -- do not modify.
        # if previous action, learn from it
        if self.prev_a is not None:
            self._rls_update(self.prev_a, obs - self.prev_obs)
        d = self.target - obs  # learn from gap to target
        # Solves and then perturbs opposite (W ~ -0.28*I thus a ~ -3.6*d)
        a = np.linalg.solve(self.W.T @ self.W + self.ridge * np.eye(2),
                            self.W.T @ d)
        n = np.linalg.norm(a)
        if n > self.action_cap:
            a *= self.action_cap / n
        # Explores and adds some noise
        eps = self.explore0 * (self.explore_decay ** self.t)
        a = np.clip(a + self.rng.normal(0, eps, size=2), self.lo, self.hi)
        self.prev_a, self.prev_obs = a.copy(), obs.copy()
        self.t += 1
        return a.astype(np.float32)

    # ===================== live: probe + ramp-hold + robust RLS =====================
    def _settled(self):
        """Settled production over the back of the hold (compensation has built)."""
        w = self.settle_window or max(1, self.hold_len // 2)
        tail = self.block_obs[-w:]
        return np.mean(np.vstack(tail), axis=0)

    def _start_block(self, ref_obs):
        if self.phase == "probe":
            self.block_action = self.probe_schedule[self.probe_idx].copy()
        else:
            self.block_action = self._control_action(ref_obs)
        self.block_obs = []
        return self.block_action

    def _fit_probe(self):
        """Least-squares per-axis gain from held probe responses: dy ~ -k * a."""
        A = np.vstack([p[0] for p in self.probe_data])   # actions  [n,2]
        DY = np.vstack([p[1] for p in self.probe_data])  # responses [n,2]
        k = np.array([self.gain_prior, self.gain_prior], dtype=float)
        for i in range(2):
            denom = float(np.sum(A[:, i] ** 2))
            if denom > 1e-6:
                k[i] = -float(np.sum(A[:, i] * DY[:, i])) / denom
        if self.k_clamp is not None:
            k = np.clip(k, self.k_clamp[0], self.k_clamp[1])
        self.W = -np.diag(k)
        self.P = [np.eye(2) * 0.5, np.eye(2) * 0.5]   # fresh covariance for control
        base = self.anchor0 if self.anchor0 is not None else self.pre_block_obs
        self.speaker_map = {"baseline": np.asarray(base, float).copy(),
                            "k": k.copy()}

    def _act_live(self, obs):
        # First call is the baseline anchor MATLAB hands over: set the reference
        # and emit the first block's action (no buffering, no update yet).
        if self.pre_block_obs is None:
            self.anchor0 = obs.copy()
            self.pre_block_obs = obs.copy()
            return self._start_block(obs).astype(np.float32)

        # Inside a hold: buffer the valid take, keep the SAME perturbation.
        self.block_obs.append(obs.copy())
        if len(self.block_obs) < self.hold_len:
            return self.block_action.astype(np.float32)

        # Block complete: measure settled response to the held action.
        settled = self._settled()
        dy = settled - self.pre_block_obs

        if self.phase == "probe":
            self.probe_data.append((self.block_action.copy(), dy.copy()))
            self.probe_idx += 1
            self.pre_block_obs = settled.copy()
            if self.probe_idx >= len(self.probe_schedule):
                self._fit_probe()        # warm-start W from the probe, switch phases
                self.phase = "control"
            return self._start_block(settled).astype(np.float32)

        # control phase: robust RLS on the settled block response, then re-aim
        if self.diagonal_only:
            self._rls_update_diag(self.block_action, dy)
        else:
            self._rls_update(self.block_action, dy)
        self._apply_k_clamp()
        self.pre_block_obs = settled.copy()
        return self._start_block(settled).astype(np.float32)

    # ---- live carryover control (used by the bridge between sessions) ----
    def reset_block_state(self, keep_estimate=True):
        """Start a fresh session. Keeps the learned W / speaker_map (warm start);
        only the within-session block bookkeeping and anchor are cleared. If a probe
        already ran, phase stays 'control' so the next session does NOT re-probe."""
        self.prev_a = self.prev_obs = None
        self.block_action = None
        self.block_obs = []
        self.pre_block_obs = None
        self.anchor0 = None
        if not keep_estimate:
            self.W = -self.gain_prior * np.eye(2)
            self.P = [np.eye(2) * 0.5, np.eye(2) * 0.5]
            self.phase = "probe" if self.probe_schedule else "control"
            self.probe_idx = 0
            self.probe_data = []
            self.speaker_map = None


if __name__ == "__main__":
    # Sim-only deps imported here so the live agent (and the MATLAB bridge) can
    # import this module without gymnasium / matplotlib installed.
    from population_sim import PopulationSpeakerEnv, run_episode_tracked
    from skeleton_loop import FixedGainAgent

    N, MAX_STEPS = 60, 30  # 60 simulated speakers, 30 trials each
    seeds = list(range(N))
    env = PopulationSpeakerEnv(max_steps=MAX_STEPS, seed=0)
    fixed = FixedGainAgent(target=env.target, gain=BEST_FIXED_GAIN)

    fix_curves, fix_final, Gs = [], [], []
    ad_curves, ad_final = [], []

    # for each seed s, runs once with the fixed and once with the adaptive
    for s in seeds:
        df, G = run_episode_tracked(env, fixed, seed=s)
        fix_curves.append(df); fix_final.append(df[-1]); Gs.append(G)
        ad = AdaptiveGEstimatorAgent(target=env.target, seed=1000 + s)
        da, _ = run_episode_tracked(env, ad, seed=s)
        ad_curves.append(da); ad_final.append(da[-1])

    fix_final = np.array(fix_final); ad_final = np.array(ad_final)

    # Prinout/Metrics
    Gs = np.array(Gs)
    cross_mag = np.sqrt(Gs[:, 0, 1] ** 2 + Gs[:, 1, 0] ** 2)
    anisotropy = np.abs(-Gs[:, 0, 0] - (-Gs[:, 1, 1]))

    fconv, ffail = fix_final < 15, fix_final > 50
    aconv, afail = ad_final < 15, ad_final > 50

    print(f"Paired comparison on {N} identical literature-grounded speakers, "
          f"{MAX_STEPS} trials:\n")
    print(f"  Fixed (gain {BEST_FIXED_GAIN}):  conv {100*fconv.mean():.0f}%   "
          f"fail {100*ffail.mean():.0f}%   mean {fix_final.mean():.1f}   "
          f"median {np.median(fix_final):.1f}")
    print(f"  Adaptive:          conv {100*aconv.mean():.0f}%   "
          f"fail {100*afail.mean():.0f}%   mean {ad_final.mean():.1f}   "
          f"median {np.median(ad_final):.1f}")

    rescued = ffail & ~afail
    print(f"\n  Speakers fixed fail: {ffail.sum()}; adaptive saved: {rescued.sum()}")
    if ffail.sum() > 0:
        print(f"  Fixed-fail tail final dist -- fixed mean {fix_final[ffail].mean():.1f}"
              f"  vs adaptive mean {ad_final[ffail].mean():.1f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    fix_mean = np.mean(np.vstack(fix_curves), axis=0)
    ad_mean = np.mean(np.vstack(ad_curves), axis=0)
    ax1.plot(fix_mean, "o-", color="tab:red", lw=2, label=f"Fixed gain {BEST_FIXED_GAIN}")
    ax1.plot(ad_mean, "o-", color="tab:blue", lw=2, label="Adaptive (G-estimator)")
    ax1.axhline(15, color="black", lw=0.6, ls="--")
    ax1.set_xlabel("Trial"); ax1.set_ylabel("Mean true distance to /u/ (60 speakers)")
    ax1.set_title("Average convergence: adaptive vs fixed (literature-grounded)")
    ax1.grid(alpha=0.3); ax1.legend(loc="best")

    sc = ax2.scatter(fix_final, ad_final, c=anisotropy, cmap="viridis",
                     s=70, edgecolor="black", lw=0.5)
    lim = max(fix_final.max(), ad_final.max(), 50) * 1.05
    ax2.plot([0, lim], [0, lim], "k--", lw=0.8)
    ax2.axhline(15, color="gray", lw=0.5); ax2.axvline(50, color="gray", lw=0.5)
    ax2.set_xlim(0, lim); ax2.set_ylim(0, lim)
    ax2.set_xlabel("Fixed-agent final distance")
    ax2.set_ylabel("Adaptive-agent final distance")
    ax2.set_title("Per-speaker: points below diagonal = adaptive wins")
    ax2.grid(alpha=0.3)
    fig.colorbar(sc, ax=ax2, label="gain anisotropy |k$_{F1}$-k$_{F2}$|")

    fig.tight_layout()
    fig.savefig("adaptive_vs_fixed.png", dpi=150)
    print("\nWrote: adaptive_vs_fixed.png")
