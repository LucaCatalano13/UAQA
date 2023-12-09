from collections import namedtuple
from dataclasses import dataclass
from random import choice, randint, random, sample
from typing import Any, List, Tuple

import numpy as np
from pandas.compat._optional import import_optional_dependency
from ..datasets.CollectionDataset import BANDS_GROUPS_IDX

# This is to allow a quick expansion of the mask from
# group-channel space into real-channel space
BAND_EXPANSION = [len(x) for x in BANDS_GROUPS_IDX.values()]

NUM_TIMESTEPS = 366 # 12

TIMESTEPS_IDX = list(range(NUM_TIMESTEPS))

MASK_STRATEGIES = (
    "group_bands",
    "random_timesteps",
    "chunk_timesteps",
    "random_combinations",
)

MaskedExample = namedtuple(
    "MaskedExample",
    ["mask_eo", "x_eo", "y_eo", "day_of_year", "day_of_week", "latlon", "strategy"],
)

def make_mask(strategy: str, mask_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Make a mask for a given strategy and percentage of masked values.
    Args:
        strategy: The masking strategy to use. One of MASK_STRATEGIES
        mask_ratio: The percentage of values to mask. Between 0 and 1.
    """

    mask = np.full((NUM_TIMESTEPS, len(BANDS_GROUPS_IDX)), False)
    num_tokens_to_mask = int(((NUM_TIMESTEPS * len(BANDS_GROUPS_IDX))) * mask_ratio)


    def random_masking(mask, num_tokens_to_mask: int):
        if num_tokens_to_mask > 0:
            # then, we flatten the mask and dw arrays
            all_tokens_mask = mask.flatten()
            unmasked_tokens = all_tokens_mask == False
            idx = np.flatnonzero(unmasked_tokens)
            np.random.shuffle(idx)
            idx = idx[:num_tokens_to_mask]
            all_tokens_mask[idx] = True
            mask = all_tokens_mask.reshape((NUM_TIMESTEPS, len(BANDS_GROUPS_IDX)))
        return mask

    # RANDOM BANDS
    if strategy == "random_combinations":
        mask = random_masking(mask, num_tokens_to_mask)

    elif strategy == "group_bands":
        # next, we figure out how many tokens we can mask
        num_band_groups_to_mask = int(num_tokens_to_mask / NUM_TIMESTEPS)
        num_tokens_to_mask -= NUM_TIMESTEPS * num_band_groups_to_mask
        assert num_tokens_to_mask >= 0
        # tuple because of mypy, which thinks lists can only hold one type
        band_groups: List[Any] = list(range(len(BANDS_GROUPS_IDX)))
        band_groups_to_mask = sample(band_groups, num_band_groups_to_mask)
        for band_group in band_groups_to_mask:
            mask[:, band_group] = True
        mask = random_masking(mask, num_tokens_to_mask)

    # RANDOM TIMESTEPS
    elif strategy == "random_timesteps":
        timesteps_to_mask = int(num_tokens_to_mask / (len(BANDS_GROUPS_IDX)))
        num_tokens_to_mask -= (len(BANDS_GROUPS_IDX)) * timesteps_to_mask
        timesteps = sample(TIMESTEPS_IDX, k=timesteps_to_mask)
        mask[timesteps] = True
        mask = random_masking(mask, num_tokens_to_mask)

    elif strategy == "chunk_timesteps":
        timesteps_to_mask = int(num_tokens_to_mask / (len(BANDS_GROUPS_IDX)))
        num_tokens_to_mask -= (len(BANDS_GROUPS_IDX)) * timesteps_to_mask
        start_idx = randint(0, NUM_TIMESTEPS - timesteps_to_mask)
        mask[start_idx : start_idx + timesteps_to_mask] = True  # noqa
        mask = random_masking(mask, num_tokens_to_mask)
    else:
        raise ValueError(f"Unknown strategy {strategy} not in {MASK_STRATEGIES}")

    return np.repeat(mask, BAND_EXPANSION, axis=1)

@dataclass
class MaskParams:
    strategies: Tuple[str, ...] = ("NDVI",)
    ratio: float = 0.5

    def __post_init__(self):
        for strategy in self.strategies:
            assert strategy in [
                "group_bands",
                "random_timesteps",
                "chunk_timesteps",
                "random_combinations",
            ]

    def mask_data(self, eo_data: np.ndarray):
        strategy = choice(self.strategies)
        mask = make_mask(strategy=strategy, mask_ratio=self.ratio)
        x = eo_data * ~mask
        y = np.zeros(eo_data.shape).astype(np.float32)
        y[mask] = eo_data[mask]
        return mask, x, y, strategy