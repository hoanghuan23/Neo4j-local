"""Compatibility imports; event consolidation lives in event_hierarchy."""

from knowledge_relations import event_hierarchy as _implementation

# Preserve historical imports, including helpers used by migration scripts.
__all__ = [name for name in vars(_implementation) if not name.startswith("__")]
globals().update({name: getattr(_implementation, name) for name in __all__})
