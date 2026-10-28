"""Enforce the user's GPU training limits before expensive allocations."""

import pytest
import torch

from pulmonary.common import read_config
from pulmonary.precision import use_amp, validate_training_limits


def test_default_budget_is_eight_images_and_at_most_twenty_epochs():
    config = read_config()
    validate_training_limits(config)
    assert config["batch_size"] == 8
    assert config["amp"] is True
    assert config["head_epochs"] + config["finetune_epochs"] == 18
    assert config["fusion_epochs"] == 20
    assert not use_amp(config, torch.device("cpu"))
    assert use_amp(config, torch.device("cuda"))


@pytest.mark.parametrize(
    "change",
    [
        {"batch_size": 16},
        {"max_epochs": 21},
        {"finetune_epochs": 20},
        {"fusion_epochs": 30},
        {"patience": 0},
    ],
)
def test_unsafe_or_unbounded_training_configuration_is_rejected(change):
    config = {**read_config(), **change}
    with pytest.raises(ValueError):
        validate_training_limits(config)
