# Compatibility wrapper.
# New code should import from world.model or world.

from world.model import WorldModel, WorldEntity, EntityObservation

__all__ = [
    "WorldModel",
    "WorldEntity",
    "EntityObservation",
]
