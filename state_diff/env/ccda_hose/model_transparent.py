from state_diff.env.ccda_hose.config import FREE_INSERT, HoseEnvConfig
from state_diff.env.ccda_hose.model import make_hose_insert_xml


def make_transparent_hose_insert_xml(
    cfg: HoseEnvConfig,
    condition: str = FREE_INSERT,
    show_occluder: bool = False,
) -> str:
    return make_hose_insert_xml(
        cfg=cfg,
        condition=condition,
        transparent_socket=True,
        show_occluder=show_occluder,
    )
