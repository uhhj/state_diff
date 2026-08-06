import copy

import numpy as np

from scripts.experiment3.phase0.occp_common import (
    build_action_script, copy_action_script)


def test_action_payload_is_deep_copied_without_recomputation():
    layout = {
        'probe_delta': np.array([0.01, 0., 0.]),
        'test_delta': np.array([0.05, 0.02, 0.]),
    }
    free = build_action_script(
        [0.5, 0., 0.02], [0., 0., 0., 1.], layout, 0.001)
    jam = copy_action_script(free)
    assert free == jam
    assert free is not jam
    jam[0]['target_position'][0] += 1
    assert free != jam
    assert free == copy.deepcopy(free)
