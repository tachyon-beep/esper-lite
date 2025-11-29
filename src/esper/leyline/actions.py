"""Leyline Actions - Action space definitions for Esper agents.

Actions represent the discrete choices available to the strategic controller.
"""

from enum import Enum


class Action(Enum):
    """Discrete actions for seed lifecycle control.

    This is the shared action space for all controllers (Tamiyo, Simic).
    Actions represent atomic decisions about seed lifecycle management.
    """
    WAIT = 0
    GERMINATE_CONV = 1      # Germinate with conv_enhance blueprint
    GERMINATE_ATTENTION = 2  # Germinate with attention blueprint
    GERMINATE_NORM = 3       # Germinate with norm blueprint
    GERMINATE_DEPTHWISE = 4  # Germinate with depthwise blueprint
    ADVANCE = 5              # Advance to next stage (training, blending, fossilize)
    CULL = 6

    @classmethod
    def is_germinate(cls, action: "Action") -> bool:
        """Check if action is any germinate variant."""
        return action in (cls.GERMINATE_CONV, cls.GERMINATE_ATTENTION,
                         cls.GERMINATE_NORM, cls.GERMINATE_DEPTHWISE)

    @classmethod
    def get_blueprint_id(cls, action: "Action") -> str | None:
        """Get blueprint ID for germinate actions, None for others."""
        return {
            cls.GERMINATE_CONV: "conv_enhance",
            cls.GERMINATE_ATTENTION: "attention",
            cls.GERMINATE_NORM: "norm",
            cls.GERMINATE_DEPTHWISE: "depthwise",
        }.get(action)


# Backwards compatibility alias (deprecated)
SimicAction = Action
