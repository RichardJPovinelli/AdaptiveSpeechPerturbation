"""Stub for skeleton_loop — provides minimal types for import only."""

from typing import Union, Sequence

import numpy as np


class FixedGainAgent:
    def __init__(
        self,
        target: Union[np.ndarray, Sequence[float]],
        gain: float = 1.0,
    ) -> None:
        self.target = np.asarray(target, dtype=np.float64)
        self.gain = gain

    def act(self, observation: Union[np.ndarray, Sequence[float]]) -> np.ndarray:
        obs = np.asarray(observation, dtype=np.float64)
        return self.gain * (self.target - obs)
