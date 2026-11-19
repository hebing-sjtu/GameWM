"""训练场 / training ground: one player with melee, ranged and dash skills, three training dummies."""

from .game import CHARACTER, SCENE, TrainingGround
from .inputs import DEMO, Bot, Face, Hold, Press, Script
from .world import Input, Snapshot, World

__all__ = ["CHARACTER", "DEMO", "SCENE", "Bot", "Face", "Hold", "Input", "Press", "Script", "Snapshot",
           "TrainingGround", "World"]
