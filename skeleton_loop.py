"""
skeleton_loop.py
================
End-to-end skeleton for the adaptive auditory-biofeedback project.

Three pieces:
  1. SpeakerEnv      -- a Gymnasium environment that simulates one speaker
                        reacting to formant-shift perturbations.
  2. FixedGainAgent  -- a deliberately dumb (non-adaptive) controller. It just
                        applies action = -gain * error. It doesn't learn G.
                        Its only job is to prove the loop works.
  3. run_episode     -- the trial-by-trial loop wiring agent and env together.

Goal of this file: confirm the entire pipeline runs and the speaker reaches
the native /u/ target. Once that's solid, we replace FixedGainAgent with the
real adaptive algorithm, and we expand SpeakerEnv into a population of varied
speakers with noise, partial compensation, and a detection threshold.

Everything operates in the Lobanov-normalized "Hz-like" frame from step 1.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------
# Targets / baseline for /u/ (from vowel_targets.csv, step 1)
# ---------------------------------------------------------------
NATIVE_U  = np.array([409.0, 1215.0])   # native Spanish /u/  (the GOAL)
EAS_U     = np.array([390.0, 1331.0])   # learners' /u/       (the BASELINE)


# ===============================================================
# 1. THE ENVIRONMENT  (the simulated speaker)
# ===============================================================
class SpeakerEnv(gym.Env):
    """Skeleton simulator: ONE hypothetical speaker, fixed response, no noise.

    State            : current produced (F1, F2)
    Action (per step): perturbation vector applied to AUDITORY FEEDBACK
                       in (F1, F2) units -- what the speaker HEARS shifted
    Observation      : the speaker's produced (F1, F2) this trial
    Reward           : -Euclidean distance from production to native target
    Episode end      : after `max_steps` trials (truncated)

    The speaker's response model is the linear-gain matrix G we discussed:

        Delta_production  =  G @ perturbation

    With G = -k * I (k > 0), the speaker compensates in the OPPOSITE direction
    of the perturbation, partially (k < 1) -- the textbook auditory-feedback
    response. Each trial's compensation accumulates in the production state,
    so adaptation persists -- this is the "learning" aftereffect.
    """

    metadata = {"render_modes": []}

    def __init__(self,
                 target=NATIVE_U,
                 baseline=EAS_U,
                 compensation_gain=0.3,    # k: partial-compensation fraction
                 max_steps=50):
        super().__init__()
        # Action: perturbation in (F1, F2). Wide bounds for now, narrowed later
        # when we add a detection threshold.
        self.action_space = spaces.Box(
            low=np.array([-200., -200.], dtype=np.float32),
            high=np.array([ 200.,  200.], dtype=np.float32),
        )
        # Observation: produced (F1, F2). Wide-ish bounds in the normalized frame.
        self.observation_space = spaces.Box(
            low=np.array([   0.,    0.], dtype=np.float32),
            high=np.array([2000., 3000.], dtype=np.float32),
        )

        self.target    = np.asarray(target,  dtype=np.float64)
        self.baseline  = np.asarray(baseline, dtype=np.float64)
        self.G         = -compensation_gain * np.eye(2)   # opposite-direction, partial
        self.max_steps = max_steps

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.production = self.baseline.copy()    # speaker starts at EAS baseline
        self.t = 0
        return self.production.astype(np.float32), {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float64)
        # Speaker reacts: their production drifts by G @ perturbation each trial.
        # No noise in the skeleton -- we'll add it when we go to a population.
        self.production = self.production + self.G @ action
        self.t += 1
        dist  = float(np.linalg.norm(self.production - self.target))
        obs   = self.production.astype(np.float32)
        reward = -dist
        terminated = False
        truncated  = self.t >= self.max_steps
        info = {"distance": dist}
        return obs, reward, terminated, truncated, info


# ===============================================================
# 2. THE AGENT  (deliberately dumb -- non-adaptive)
# ===============================================================
class FixedGainAgent:
    """Non-adaptive controller. Picks perturbation purely from current error.

    Logic:
      error  = target - production           (where we WANT production to go)
      action = -gain * error                 (perturb opposite, since the
                                              speaker compensates opposite)

    Why this works (in the noiseless, fixed-G world):
      Production update : production_{t+1} = production_t + G @ action
                                           = production_t + (-k I)(-gain * error)
                                           = production_t + gain*k * error
      Error update      : error_{t+1}      = (1 - gain*k) * error_t

      So the error decays by a factor of (1 - gain*k) per step. With k=0.3
      and gain=1.5, that's a ~55% closure per step -- visible convergence
      over a handful of trials. This is just to prove the loop works; it
      does NOT learn k from data.
    """

    def __init__(self, target, gain=1.5, action_low=-200., action_high=200.):
        self.target = np.asarray(target, dtype=np.float64)
        self.gain   = gain
        self.lo, self.hi = action_low, action_high

    def act(self, observation):
        error  = self.target - np.asarray(observation, dtype=np.float64)
        action = -self.gain * error
        return np.clip(action, self.lo, self.hi).astype(np.float32)


# ===============================================================
# 3. THE LOOP  (one episode = one simulated session)
# ===============================================================
def run_episode(env, agent):
    """Run one episode; return per-trial trajectories of production, action, dist."""
    obs, _ = env.reset()
    productions = [obs.copy()]
    actions     = []
    distances   = [float(np.linalg.norm(obs - env.target))]
    while True:
        action = agent.act(obs)
        actions.append(action.copy())
        obs, reward, terminated, truncated, info = env.step(action)
        productions.append(obs.copy())
        distances.append(info["distance"])
        if terminated or truncated:
            break
    return np.array(productions), np.array(actions), np.array(distances)


# ===============================================================
# 4. DEMO  (run it and plot)
# ===============================================================
if __name__ == "__main__":
    env   = SpeakerEnv(max_steps=20)
    agent = FixedGainAgent(target=env.target, gain=1.5)
    productions, actions, distances = run_episode(env, agent)

    print(f"Start  : F1={productions[0,0]:7.1f}  F2={productions[0,1]:7.1f}  "
          f"dist={distances[0]:6.1f}")
    print(f"End    : F1={productions[-1,0]:7.1f}  F2={productions[-1,1]:7.1f}  "
          f"dist={distances[-1]:6.1f}")
    print(f"Target : F1={env.target[0]:7.1f}  F2={env.target[1]:7.1f}")
    print(f"Trials : {len(actions)}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: trajectory in F1-F2 space, phonetic-chart conventions
    ax1.plot(productions[:, 1], productions[:, 0], "o-", color="purple",
             markersize=5, linewidth=1.2, label="Production trajectory")
    ax1.scatter(env.baseline[1], env.baseline[0], s=120, marker="s",
                color="tab:red", zorder=4, label="Baseline (EAS /u/)")
    ax1.scatter(env.target[1],   env.target[0],   s=140, marker="*",
                color="tab:blue", zorder=4, label="Target (Native /u/)")
    ax1.set_xlabel("F2 (normalized, Hz-like)")
    ax1.set_ylabel("F1 (normalized, Hz-like)")
    ax1.set_title("Production trajectory through F1-F2 space")
    ax1.invert_xaxis(); ax1.invert_yaxis()
    ax1.grid(alpha=0.3); ax1.legend(loc="best")

    # Right: distance-to-target vs trial number
    ax2.plot(distances, "o-", color="purple")
    ax2.axhline(0, color="black", lw=0.5)
    ax2.set_xlabel("Trial")
    ax2.set_ylabel("Distance to native /u/")
    ax2.set_title("Convergence curve")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig("skeleton_convergence.png", dpi=150)
    print("\nWrote: skeleton_convergence.png")
