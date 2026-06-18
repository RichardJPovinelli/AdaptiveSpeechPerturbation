"""Stub for population_sim — provides minimal types for import only."""

from typing import Tuple, List

import numpy as np


class PopulationSpeakerEnv:
    def __init__(self, max_steps: int = 30, seed: int = 0) -> None:
        self.max_steps = max_steps
        self.seed = seed
        self.target = np.array([300.0, 3000.0])


def run_episode_tracked(
    env: PopulationSpeakerEnv, agent: object, seed: int = 0
) -> Tuple[List[float], np.ndarray]:
    """Return dummy distance curve and G matrix."""
    return [0.0], np.zeros((2, 2))
