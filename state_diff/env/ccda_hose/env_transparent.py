from typing import Optional

from state_diff.env.ccda_hose.config import FREE_INSERT, HoseEnvConfig
from state_diff.env.ccda_hose.env import HiddenJamHoseInsertionEnv, scripted_rollout


class TransparentSocketHoseInsertionEnv(HiddenJamHoseInsertionEnv):
    def __init__(
        self,
        config: Optional[HoseEnvConfig] = None,
        condition: str = FREE_INSERT,
        seed: int = 0,
        show_occluder: bool = False,
    ) -> None:
        super().__init__(
            config=config,
            condition=condition,
            seed=seed,
            transparent_socket=True,
            show_occluder=show_occluder,
        )


def transparent_scripted_rollout(
    condition: str = FREE_INSERT,
    seed: int = 0,
    config: Optional[HoseEnvConfig] = None,
    record_frames: bool = True,
    camera_name: str = "side_top",
    show_occluder: bool = False,
):
    return scripted_rollout(
        condition=condition,
        seed=seed,
        config=config,
        record_frames=record_frames,
        camera_name=camera_name,
        transparent_socket=True,
        show_occluder=show_occluder,
    )
