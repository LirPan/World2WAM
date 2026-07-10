"""Version B: Physics-aligned World2WAM on FastWAM MoT action tokens."""

from world2wam_vb.models import (
    BidirectionalMotWorld2WAM,
    PhysicsAlignedMotWorld2WAM,
    build_bidirectional_mot_model,
    build_physics_mot_model,
)

__version__ = "0.1.0"
__all__ = [
    "BidirectionalMotWorld2WAM",
    "PhysicsAlignedMotWorld2WAM",
    "build_bidirectional_mot_model",
    "build_physics_mot_model",
]
