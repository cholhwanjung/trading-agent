"""Interaction Layer — Chat Gateway·토론·브리핑."""

from interaction.briefing import build_briefing, write_briefing
from interaction.chat import ChatAnswer, ChatEngine, GroundingError, enforce_grounding
from interaction.context import allowed_ids, build_context

__all__ = [
    "ChatAnswer",
    "ChatEngine",
    "GroundingError",
    "allowed_ids",
    "build_briefing",
    "build_context",
    "enforce_grounding",
    "write_briefing",
]
