"""Enforced training budget for the team's 6 GB GTX 1660 SUPER."""

import torch


def validate_training_limits(config):
    if config["batch_size"] != 8:
        raise ValueError("This project requires batch_size=8 for the 6 GB GPU")
    cap = config.get("max_epochs", 20)
    if not isinstance(cap, int) or not 1 <= cap <= 20:
        raise ValueError("max_epochs must be an integer between 1 and 20")
    if min(config["head_epochs"], config["finetune_epochs"]) < 0:
        raise ValueError("Epoch counts cannot be negative")
    if not 1 <= config["head_epochs"] + config["finetune_epochs"] <= cap:
        raise ValueError("Head + fine-tuning epochs must total no more than max_epochs")
    if not 1 <= config["fusion_epochs"] <= cap:
        raise ValueError("Fusion epochs must not exceed max_epochs")
    if not 1 <= config["patience"] <= cap:
        raise ValueError("Early stopping patience must be between 1 and max_epochs")


def use_amp(config, device):
    return bool(config.get("amp", True)) and device.type == "cuda"


def grad_scaler(enabled):
    # A conservative initial scale reduces warm-up overflows on older GPUs.
    return torch.amp.GradScaler("cuda", enabled=enabled, init_scale=1024.0)
