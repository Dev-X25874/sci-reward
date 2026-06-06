"""sci-reward: JAX-native scientific reward modeling toolkit."""

from sci_reward.rewards.chemical import QEDReward, SAScoreReward, ValiditySMILES
from sci_reward.rewards.composite import CompositeReward
from sci_reward.rewards.format import IUPACFormatReward, SMILESFormatReward

__version__ = "0.1.0"
__all__ = [
    "ValiditySMILES",
    "QEDReward",
    "SAScoreReward",
    "SMILESFormatReward",
    "IUPACFormatReward",
    "CompositeReward",
]
