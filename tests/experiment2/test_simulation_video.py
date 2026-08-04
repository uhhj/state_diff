import numpy as np

from scripts.experiment2.phase0.simulation_video_utils import compose_frame, letterbox


def test_letterbox_returns_requested_shape():
    frame = np.full((40, 80, 3), 100, dtype=np.uint8)
    output = letterbox(frame, width=120, height=90)
    assert output.shape == (90, 120, 3)
    assert output.dtype == np.uint8


def test_compose_frame_places_two_panels_and_header():
    free = np.full((40, 80, 3), 50, dtype=np.uint8)
    hidden = np.full((60, 50, 3), 180, dtype=np.uint8)
    output = compose_frame(
        free,
        hidden,
        panel_width=100,
        panel_height=80,
        group_id="hf_070001",
        action_hash="a" * 64,
    )
    assert output.shape == (122, 200, 3)
    assert output.dtype == np.uint8
