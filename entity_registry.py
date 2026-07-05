# Compatibility wrapper.
# New code should import from world.entity_registry or world.

from world.entity_registry import EntityRegistry

__all__ = [
    "EntityRegistry",
]
