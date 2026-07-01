from dataclasses import dataclass
from typing import Tuple


@dataclass
class HoseEnvConfig:
    """Configuration for Hidden Lateral-Jam Hose Insertion.

    Units follow MuJoCo SI convention: meter, kilogram, second.
    The defaults are intentionally conservative for stable qualitative
    CCDA auditing rather than high-fidelity soft-body simulation.
    """

    # Hose geometry
    n_segments: int = 14
    segment_length: float = 0.018
    hose_radius: float = 0.005
    hose_density: float = 850.0
    hose_joint_damping: float = 0.035
    hose_joint_stiffness: float = 0.18

    # Plug/head geometry
    plug_length: float = 0.024
    plug_radius: float = 0.0056
    plug_density: float = 1200.0

    # Socket geometry
    socket_depth: float = 0.13
    socket_half_width: float = 0.0095
    socket_wall_thickness: float = 0.008
    socket_center_z: float = 0.045
    socket_entrance_x: float = 0.0

    # Hidden jam geometry
    jam_block_x: float = 0.065
    jam_block_y: float = 0.0050
    jam_block_z: float = 0.045
    jam_block_size: Tuple[float, float, float] = (0.050, 0.0045, 0.008)
    jam_friction: Tuple[float, float, float] = (8.0, 0.16, 0.04)

    # Simulation
    timestep: float = 0.004
    frame_skip: int = 5
    solver_iterations: int = 80
    settle_steps: int = 35

    # Scripted controller
    start_x: float = -0.055
    pre_insert_x: float = -0.016
    push_distance: float = 0.115
    approach_steps: int = 45
    push_steps: int = 90
    max_delta_per_env_step: float = 0.0022

    # Randomization for small-scale debug dataset
    pos_noise: float = 0.0015
    y_noise: float = 0.0008
    z_noise: float = 0.0008

    # Observation
    visible_segments: int = 7

    # Labels and thresholds
    success_insert_depth: float = 0.085
    half_insert_depth: float = 0.035
    lateral_offset_threshold: float = 0.010
    max_curvature_success_threshold: float = 38.0
    max_curvature_buckle_threshold: float = 58.0
    jam_force_threshold: float = 0.12

    # Rendering
    render_width: int = 640
    render_height: int = 480


FREE_INSERT = "free_insert"
RIGHT_HIDDEN_JAM = "right_hidden_jam"
SUPPORTED_CONDITIONS = (FREE_INSERT, RIGHT_HIDDEN_JAM)
