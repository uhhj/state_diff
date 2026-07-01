from __future__ import annotations

from xml.sax.saxutils import escape

from state_diff.env.ccda_hose.config import (
    FREE_INSERT,
    RIGHT_HIDDEN_JAM,
    HoseEnvConfig,
    SUPPORTED_CONDITIONS,
)


def _fmt(x: float) -> str:
    return f"{float(x):.8g}"


def _vec(*values: float) -> str:
    return " ".join(_fmt(v) for v in values)


def _make_hose_chain_xml(cfg: HoseEnvConfig) -> str:
    """Create nested MuJoCo bodies for a segmented hose.

    The plug is a free body welded to a mocap gripper. Hose segments extend
    backward along negative x. Each segment is connected by a ball joint.
    """
    seg_len = cfg.segment_length
    r = cfg.hose_radius

    lines = []
    indent = "      "
    first_pos = -cfg.plug_length / 2.0
    for i in range(cfg.n_segments):
        pos_x = first_pos if i == 0 else -seg_len
        lines.append(f'{indent}<body name="hose_seg_{i}" pos="{_vec(pos_x, 0, 0)}">')
        lines.append(
            f'{indent}  <joint name="hose_joint_{i}" type="ball" '
            f'damping="{_fmt(cfg.hose_joint_damping)}" '
            f'stiffness="{_fmt(cfg.hose_joint_stiffness)}" armature="0.0005"/>'
        )
        lines.append(
            f'{indent}  <geom name="hose_geom_{i}" type="capsule" '
            f'fromto="{_vec(0, 0, 0, -seg_len, 0, 0)}" '
            f'size="{_fmt(r)}" density="{_fmt(cfg.hose_density)}" '
            f'material="hose_mat" friction="1.1 0.03 0.01" '
            f'solref="0.012 1" solimp="0.88 0.95 0.001"/>'
        )
        indent += "  "

    for i in reversed(range(cfg.n_segments)):
        indent = "      " + "  " * i
        lines.append(f"{indent}</body>")
    return "\n".join(lines)


def make_hose_insert_xml(
    cfg: HoseEnvConfig,
    condition: str = FREE_INSERT,
    transparent_socket: bool = False,
    show_occluder: bool = True,
) -> str:
    if condition not in SUPPORTED_CONDITIONS:
        raise ValueError(f"Unsupported condition {condition!r}. Expected one of {SUPPORTED_CONDITIONS}.")

    half = cfg.socket_half_width
    thick = cfg.socket_wall_thickness
    depth = cfg.socket_depth
    zc = cfg.socket_center_z
    x_center = cfg.socket_entrance_x + depth / 2.0

    socket_rgba = "0.05 0.05 0.05 1"
    socket_collision_rgba = "0.05 0.05 0.05 0.04" if transparent_socket else socket_rgba
    socket_visual_rgba = "0.2 0.8 1.0 0.20"
    occluder_rgba = "0.02 0.02 0.02 0.08" if transparent_socket else "0.02 0.02 0.02 1"
    jam_rgba = "1.0 0.05 0.05 0.55" if transparent_socket else "0.85 0.05 0.05 1"

    jam_xml = ""
    if condition == RIGHT_HIDDEN_JAM:
        sx, sy, sz = cfg.jam_block_size
        fx, fy, fz = cfg.jam_friction
        jam_xml = f"""
    <geom name="hidden_jam_block" type="box"
          pos="{_vec(cfg.jam_block_x, cfg.jam_block_y, cfg.jam_block_z)}"
          size="{_vec(sx, sy, sz)}"
          material="jam_mat"
          rgba="{jam_rgba}"
          friction="{_vec(fx, fy, fz)}"
          solref="0.006 1" solimp="0.94 0.98 0.001"/>
"""

    occluder_xml = ""
    if show_occluder:
        occluder_xml = f"""
    <geom name="front_occluder_right" type="box"
          pos="{_vec(-0.002, half + thick * 0.45, zc)}"
          size="{_vec(0.0025, thick * 0.65, half + thick)}"
          material="occluder_mat" rgba="{occluder_rgba}" contype="0" conaffinity="0"/>
    <geom name="front_occluder_left" type="box"
          pos="{_vec(-0.002, -half - thick * 0.45, zc)}"
          size="{_vec(0.0025, thick * 0.65, half + thick)}"
          material="occluder_mat" rgba="{occluder_rgba}" contype="0" conaffinity="0"/>
"""

    socket_visual_xml = ""
    if transparent_socket:
        socket_visual_xml = f"""
    <geom name="socket_visual_right" type="box"
          pos="{_vec(x_center, half + thick / 2.0, zc)}"
          size="{_vec(depth / 2.0, thick / 2.0, half + thick)}"
          rgba="{socket_visual_rgba}" contype="0" conaffinity="0"/>
    <geom name="socket_visual_left" type="box"
          pos="{_vec(x_center, -half - thick / 2.0, zc)}"
          size="{_vec(depth / 2.0, thick / 2.0, half + thick)}"
          rgba="{socket_visual_rgba}" contype="0" conaffinity="0"/>
    <geom name="socket_visual_top" type="box"
          pos="{_vec(x_center, 0, zc + half + thick / 2.0)}"
          size="{_vec(depth / 2.0, half + thick, thick / 2.0)}"
          rgba="{socket_visual_rgba}" contype="0" conaffinity="0"/>
    <geom name="socket_visual_bottom" type="box"
          pos="{_vec(x_center, 0, zc - half - thick / 2.0)}"
          size="{_vec(depth / 2.0, half + thick, thick / 2.0)}"
          rgba="{socket_visual_rgba}" contype="0" conaffinity="0"/>
"""

    hose_chain_xml = _make_hose_chain_xml(cfg)

    xml = f"""
<mujoco model="ccda_hidden_lateral_jam_hose_insertion_{escape(condition)}">
  <compiler angle="radian" inertiafromgeom="true"/>
  <option timestep="{_fmt(cfg.timestep)}" gravity="0 0 -9.81"
          integrator="implicit" cone="elliptic"
          iterations="{cfg.solver_iterations}" tolerance="1e-9"/>

  <default>
    <geom contype="1" conaffinity="1" rgba="0.7 0.7 0.7 1"/>
    <joint limited="false"/>
  </default>

  <asset>
    <material name="table_mat" rgba="0.32 0.32 0.32 1"/>
    <material name="socket_mat" rgba="{socket_rgba}"/>
    <material name="hose_mat" rgba="0.05 0.35 0.85 1"/>
    <material name="plug_mat" rgba="0.95 0.65 0.15 1"/>
    <material name="gripper_mat" rgba="0.10 0.10 0.10 0.45"/>
    <material name="jam_mat" rgba="{jam_rgba}"/>
    <material name="occluder_mat" rgba="{occluder_rgba}"/>
  </asset>

  <worldbody>
    <light name="key_light" pos="-0.2 -0.45 0.55" dir="0.4 0.7 -1" diffuse="0.8 0.8 0.8"/>
    <camera name="front" pos="0.035 -0.43 0.09" xyaxes="1 0 0 0 0 1" fovy="42"/>
    <camera name="top" pos="0.045 0.0 0.45" xyaxes="1 0 0 0 1 0" fovy="45"/>
    <camera name="side" pos="-0.18 -0.20 0.11" xyaxes="0.7 -0.7 0 0 0 1" fovy="45"/>
    <camera name="side_top" pos="-0.10 -0.34 0.24" xyaxes="0.96 -0.28 0 0.18 0.62 0.76" fovy="36"/>
    <camera name="debug_close" pos="-0.035 -0.20 0.135" xyaxes="0.93 -0.36 0 0.16 0.42 0.89" fovy="32"/>

    <geom name="table" type="box" pos="{_vec(0.025, 0, -0.006)}"
          size="{_vec(0.30, 0.20, 0.006)}" material="table_mat"
          friction="1.1 0.02 0.01"/>

    <geom name="socket_wall_right" type="box"
          pos="{_vec(x_center, half + thick / 2.0, zc)}"
          size="{_vec(depth / 2.0, thick / 2.0, half + thick)}"
          material="socket_mat" rgba="{socket_collision_rgba}" contype="1" conaffinity="1"
          friction="0.75 0.02 0.005"/>
    <geom name="socket_wall_left" type="box"
          pos="{_vec(x_center, -half - thick / 2.0, zc)}"
          size="{_vec(depth / 2.0, thick / 2.0, half + thick)}"
          material="socket_mat" rgba="{socket_collision_rgba}" contype="1" conaffinity="1"
          friction="0.75 0.02 0.005"/>
    <geom name="socket_wall_top" type="box"
          pos="{_vec(x_center, 0, zc + half + thick / 2.0)}"
          size="{_vec(depth / 2.0, half + thick, thick / 2.0)}"
          material="socket_mat" rgba="{socket_collision_rgba}" contype="1" conaffinity="1"
          friction="0.75 0.02 0.005"/>
    <geom name="socket_wall_bottom" type="box"
          pos="{_vec(x_center, 0, zc - half - thick / 2.0)}"
          size="{_vec(depth / 2.0, half + thick, thick / 2.0)}"
          material="socket_mat" rgba="{socket_collision_rgba}" contype="1" conaffinity="1"
          friction="0.75 0.02 0.005"/>

{socket_visual_xml}

{occluder_xml}
{jam_xml}

    <body name="gripper_mocap" mocap="true" pos="{_vec(cfg.start_x, 0, zc)}">
      <geom name="gripper_left_finger" type="box"
            pos="{_vec(-0.004, 0.012, 0)}" size="0.014 0.002 0.010"
            material="gripper_mat" contype="0" conaffinity="0"/>
      <geom name="gripper_right_finger" type="box"
            pos="{_vec(-0.004, -0.012, 0)}" size="0.014 0.002 0.010"
            material="gripper_mat" contype="0" conaffinity="0"/>
    </body>

    <body name="plug" pos="{_vec(cfg.start_x, 0, zc)}">
      <freejoint name="plug_free"/>
      <geom name="plug_geom" type="capsule"
            fromto="{_vec(-cfg.plug_length / 2.0, 0, 0, cfg.plug_length / 2.0, 0, 0)}"
            size="{_fmt(cfg.plug_radius)}" density="{_fmt(cfg.plug_density)}"
            material="plug_mat" friction="1.05 0.04 0.01"
            solref="0.008 1" solimp="0.9 0.96 0.001"/>
{hose_chain_xml}
    </body>
  </worldbody>

  <equality>
    <weld name="grip_to_plug" body1="gripper_mocap" body2="plug"
          solref="0.006 1" solimp="0.95 0.99 0.001"/>
  </equality>
</mujoco>
"""
    return xml
