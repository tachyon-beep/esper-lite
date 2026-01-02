from __future__ import annotations

import torch

from esper.kasmina.blending import BlendAlgorithm


class _CachedAlphaBlend(BlendAlgorithm):
    algorithm_id = "cached_alpha"

    def __init__(self) -> None:
        super().__init__()
        self.total_steps = 1

    def get_alpha_for_blend(self, x: torch.Tensor) -> torch.Tensor:
        return self._get_cached_alpha_tensor(0.5, x)


def test_thread_local_cache_has_default_value():
    blend = _CachedAlphaBlend()
    assert blend._alpha_cache_local.cache is None


def test_thread_local_cache_reuses_tensor_when_unchanged():
    blend = _CachedAlphaBlend()
    x = torch.zeros((2, 3), dtype=torch.float32)

    alpha_1 = blend.get_alpha_for_blend(x)
    alpha_2 = blend.get_alpha_for_blend(x)
    assert alpha_1 is alpha_2

    alpha_3 = blend._get_cached_alpha_tensor(0.6, x)
    assert alpha_3 is not alpha_1

    blend.reset_cache()
    alpha_4 = blend.get_alpha_for_blend(x)
    assert alpha_4 is not alpha_1
