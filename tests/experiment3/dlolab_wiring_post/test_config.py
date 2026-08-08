import json
from pathlib import Path


ROOT = (
    Path(__file__)
    .resolve()
    .parents[3]
)


def test_pb0_config_freezes_published_physics():
    config = json.loads(
        (
            ROOT
            / "configs"
            / "experiment3"
            / "published_benchmark"
            / "dlolab_wiring_post_pb0.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    benchmark = config[
        "benchmark"
    ]

    assert benchmark[
        "task"
    ] == "wiring_post"

    assert benchmark[
        "modify_benchmark_physics"
    ] is False

    assert benchmark[
        "modify_reward"
    ] is False

    assert benchmark[
        "modify_geometry"
    ] is False

    assert config[
        "pair_mining"
    ][
        "scaleup_batches"
    ] == 16
