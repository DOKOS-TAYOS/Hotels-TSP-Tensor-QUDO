"""Instance generation utilities for baseline experimentation."""

from __future__ import annotations

import random

from instance_gen_process.models import CargoItem, InstanceConfig, ProblemInstance


def generate_random_instance(config: InstanceConfig) -> ProblemInstance:
    """Generate a random `ProblemInstance` from `InstanceConfig` ranges."""

    rng = random.Random(config.seed)
    items = []
    for item_id in range(config.num_items):
        weight = rng.uniform(*config.weight_range)
        volume = rng.uniform(*config.volume_range)
        items.append(CargoItem(item_id=item_id, weight=weight, volume=volume))

    return ProblemInstance(
        items=tuple(items),
        max_weight=config.max_weight,
        max_volume=config.max_volume,
        cg_min=config.cg_min,
        cg_max=config.cg_max,
    )

