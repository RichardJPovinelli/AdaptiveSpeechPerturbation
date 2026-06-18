import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from skeleton_loop import FixedGainAgent, EAS_U, NATIVE_U

EAS_U_EMPIRICAL   = [390.4, 1330.1]
NATIVE_U_EMPIRICAL = [408.9, 1213.9]

class PopulationSpeakerEnv(gym.Env):

    metadata = {"render_modes": []}

    def __init__(self,
                 target=NATIVE_U, baseline=EAS_U, max_steps=30,
                 # per-formant gains (MacDonald 2010 slope table)
                 gain_F1_mean=0.26, gain_F2_mean=0.30,
                 gain_F1_std=0.18,  gain_F2_std=0.26,
                 gain_F1_F2_corr=0.50,         # MacDonald 2011 meta-analysis
                 # Dont cut off the tail paper - include some negative possibilities 
                 gain_floor=-0.20, gain_ceil=0.65,
                 # cross-coupling (small; MacDonald 2011 Exp 1)
                 cross_coupling_std=0.04,
                 # nonlinearity
                 saturation_amount=70.0,
                 detection_threshold=220.0, detection_softness=25.0,
                 # measurement noise (Lametti-style baseline scatter)
                 noise_std=12.0,
                 # voluntary articulatory drift (OFF by default -- see step()).
                 # drift_std=0.0 reproduces the pre-drift results bit-for-bit
                 # because the rng is only touched when drift is enabled.
                 drift_std=0.0, drift_jump_prob=0.0, drift_jump_std=0.0,
                 seed=None):
        super().__init__()
        self.action_space = spaces.Box(
            low=np.array([-200., -200.], dtype=np.float32),
            high=np.array([ 200.,  200.], dtype=np.float32),
        )
        self.observation_space = spaces.Box(
            low=np.array([   0.,    0.], dtype=np.float32),
            high=np.array([2000., 3000.], dtype=np.float32),
        )
        self.target   = np.asarray(target,   dtype=np.float64)
        self.baseline = np.asarray(baseline, dtype=np.float64)
        self.max_steps = max_steps

        # gain distribution
        self.gain_F1_mean = gain_F1_mean
        self.gain_F2_mean = gain_F2_mean
        self.gain_F1_std  = gain_F1_std
        self.gain_F2_std  = gain_F2_std
        self.gain_corr    = gain_F1_F2_corr
        self.gain_floor   = gain_floor
        self.gain_ceil    = gain_ceil
        self.cross_coupling_std = cross_coupling_std

        # nonlinearity
        self.saturation = saturation_amount
        self.det_th     = detection_threshold
        self.det_soft   = detection_softness

        # noise
        self.noise_std  = noise_std

        # voluntary articulatory drift -- a wander of the PRODUCTION state that
        # the controller did not cause (changing mouth shape, posture, fatigue).
        self.drift_std       = drift_std
        self.drift_jump_prob = drift_jump_prob
        self.drift_jump_std  = drift_jump_std

        self._rng = np.random.default_rng(seed)

    def _sample_speaker(self):
        # Sample one speaker's response matrix G.
        
        mean = np.array([self.gain_F1_mean, self.gain_F2_mean])
        s1, s2 = self.gain_F1_std, self.gain_F2_std
        cov  = np.array([[s1 * s1,             self.gain_corr * s1 * s2],
                         [self.gain_corr * s1 * s2, s2 * s2]])
        k = self._rng.multivariate_normal(mean, cov)
        k = np.clip(k, self.gain_floor, self.gain_ceil)
        G = -np.diag(k)
        G[0, 1] = self._rng.normal(0, self.cross_coupling_std)
        G[1, 0] = self._rng.normal(0, self.cross_coupling_std)
        return G, k

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.G, self.k = self._sample_speaker()
        self.production = self.baseline.copy()
        self.t = 0
        obs = self.production + self._rng.normal(0, self.noise_std, size=2)
        return obs.astype(np.float32), {"G": self.G.copy(), "k": self.k.copy()}

    def step(self, action):
        action = np.asarray(action, dtype=np.float64)
        raw_delta = self.G @ action
        delta = self.saturation * np.tanh(raw_delta / self.saturation)
        a_mag = np.linalg.norm(action)
        gate  = 1.0 / (1.0 + np.exp((a_mag - self.det_th) / self.det_soft))
        delta = gate * delta

        # --- voluntary articulatory drift ---
        # Added to the PRODUCTION STATE (not the observation), so it accumulates
        # and persists like a real articulatory change -- and so an online
        # estimator misreads (production_t - production_{t-1}) as the effect of
        # its own last action. This is exactly the failure you saw self-testing:
        # the RLS attributed your mouth-shape change to its perturbation and the
        # gain/cross-term estimate blew up. A pure random walk (no mean reversion)
        # is the honest worst case for identification; jumps model posture shifts.
        # Guarded by drift_std > 0 so the rng stream is untouched when drift is off.
        drift = np.zeros(2)
        if self.drift_std > 0.0:
            drift = self._rng.normal(0, self.drift_std, size=2)
            if self.drift_jump_prob > 0.0 and self._rng.random() < self.drift_jump_prob:
                drift = drift + self._rng.normal(0, self.drift_jump_std, size=2)
        self.production = self.production + delta + drift
        obs = self.production + self._rng.normal(0, self.noise_std, size=2)
        self.t += 1
        true_dist = float(np.linalg.norm(self.production - self.target))
        reward    = -true_dist
        terminated = False
        truncated  = self.t >= self.max_steps
        info = {"true_distance": true_dist, "gate": float(gate),
                "drift": float(np.linalg.norm(drift))}
        return obs.astype(np.float32), reward, terminated, truncated, info


def run_episode_tracked(env, agent, seed):
    obs, info = env.reset(seed=seed)
    G = info["G"].copy()
    true_dists = [float(np.linalg.norm(env.production - env.target))]
    while True:
        action = agent.act(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        true_dists.append(info["true_distance"])
        if terminated or truncated:
            break
    return np.array(true_dists), G


def population_final_distances(env, gain, seeds):
    agent = FixedGainAgent(target=env.target, gain=gain)
    finals = []
    for s in seeds:
        d, _ = run_episode_tracked(env, agent, seed=s)
        finals.append(d[-1])
    return np.array(finals)


if __name__ == "__main__":
    N_SPEAKERS = 60
    MAX_STEPS  = 30
    seeds = list(range(N_SPEAKERS))
    env   = PopulationSpeakerEnv(max_steps=MAX_STEPS, seed=0)

    # ---- 1. Tune the fixed gain to the population optimum (fair baseline) ----
    gain_grid = np.round(np.arange(0.5, 4.01, 0.25), 2)
    sweep = []
    for g in gain_grid:
        finals = population_final_distances(env, g, seeds)
        sweep.append((g, np.median(finals), finals.mean()))
    sweep = np.array(sweep)
    best_idx  = int(np.argmin(sweep[:, 2]))   # by population MEAN
    best_gain = float(sweep[best_idx, 0])

    print("=== Fixed-gain sweep (median | mean final distance) ===")
    for g, med, mean in sweep:
        star = "  <-- best" if g == best_gain else ""
        print(f"  gain={g:4.2f}   median={med:6.1f}   mean={mean:6.1f}{star}")
    print(f"\nChosen best fixed gain: {best_gain}\n")

    # ---- 2. Full run with the best fixed gain ----
    agent = FixedGainAgent(target=env.target, gain=best_gain)
    all_dists, final_dist, Gs = [], [], []
    for s in seeds:
        d, G = run_episode_tracked(env, agent, seed=s)
        all_dists.append(d); final_dist.append(d[-1]); Gs.append(G)
    final_dist = np.array(final_dist)
    Gs = np.array(Gs)

    converged = final_dist < 15
    failed    = final_dist > 50
    struggled = ~(converged | failed)

    print(f"Best non-adaptive baseline on {N_SPEAKERS} speakers, {MAX_STEPS} trials each:")
    print(f"  Converged (<15): {converged.sum()}  ({100*converged.mean():.0f}%)")
    print(f"  Struggled (15-50): {struggled.sum()}  ({100*struggled.mean():.0f}%)")
    print(f"  Failed    (>50): {failed.sum()}  ({100*failed.mean():.0f}%)")
    print(f"  Final distance -- mean={final_dist.mean():.1f}, "
          f"median={np.median(final_dist):.1f}, max={final_dist.max():.1f}")

    k_f1 = -Gs[:, 0, 0]
    k_f2 = -Gs[:, 1, 1]
    anisotropy = np.abs(k_f1 - k_f2)
    cross_mag  = np.sqrt(Gs[:, 0, 1] ** 2 + Gs[:, 1, 0] ** 2)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    for d, ok, bad in zip(all_dists, converged, failed):
        if ok:    color, alpha = "tab:green",  0.45
        elif bad: color, alpha = "tab:red",    0.7
        else:     color, alpha = "tab:orange", 0.6
        ax1.plot(d, color=color, alpha=alpha, lw=1.1)
    ax1.axhline(15, color="black", lw=0.6, ls="--")
    ax1.set_xlabel("Trial"); ax1.set_ylabel("True distance to native /u/")
    ax1.set_title(f"Best fixed gain ({best_gain}) on {N_SPEAKERS} speakers")
    ax1.grid(alpha=0.3)
    from matplotlib.lines import Line2D
    ax1.legend(handles=[
        Line2D([0],[0], color="tab:green",  lw=2, label=f"Converged (<15): {converged.sum()}"),
        Line2D([0],[0], color="tab:orange", lw=2, label=f"Struggled (15-50): {struggled.sum()}"),
        Line2D([0],[0], color="tab:red",    lw=2, label=f"Failed (>50): {failed.sum()}"),
    ], loc="best")

    ax2.plot(sweep[:, 0], sweep[:, 1], "o-", label="median final dist")
    ax2.plot(sweep[:, 0], sweep[:, 2], "s--", color="gray", label="mean final dist")
    ax2.axvline(best_gain, color="tab:blue", lw=1, ls=":")
    ax2.set_xlabel("Fixed gain"); ax2.set_ylabel("Final distance over population")
    ax2.set_title("No single gain serves everyone")
    ax2.grid(alpha=0.3); ax2.legend(loc="best")

    sc = ax3.scatter(anisotropy, k_f2, c=final_dist, cmap="RdYlGn_r",
                     s=70, edgecolor="black", lw=0.5, vmin=0, vmax=80)
    ax3.set_xlabel("Gain anisotropy  |k$_{F1}$ - k$_{F2}$|")
    ax3.set_ylabel("Speaker's F2 gain  k$_{F2}$")
    ax3.set_title("Where the fixed agent fails (red)")
    ax3.grid(alpha=0.3)
    fig.colorbar(sc, ax=ax3, label="final distance")

    fig.tight_layout()
    fig.savefig("population_sim.png", dpi=150)
    print("\nWrote: population_sim.png")