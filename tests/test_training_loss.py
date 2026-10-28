"""Weighted validation loss must be independent of batch class composition."""

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from pulmonary.image_models import train_epoch


def test_weighted_epoch_loss_matches_whole_dataset_loss():
    logits = torch.tensor([[3.0, -1.0], [0.0, 1.0], [2.0, -1.0], [3.0, -2.0], [-1.0, 1.0]])
    labels = torch.tensor([0, 0, 0, 1, 1])
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 10.0]))
    loader = DataLoader(TensorDataset(logits, labels), batch_size=2)
    result = train_epoch(nn.Identity(), loader, criterion, torch.device("cpu"))
    assert result == pytest.approx(float(criterion(logits, labels)), rel=1e-6)


def test_amp_forward_overflow_retries_fp32_and_records_event():
    class OverflowOnlyInAutocast(nn.Module):
        def forward(self, values):
            return values * float("inf") if torch.is_autocast_enabled("cpu") else values

    values = torch.tensor([[1.0, 2.0], [2.0, 1.0]])
    targets = torch.tensor([1, 0])
    loader = DataLoader(TensorDataset(values, targets), batch_size=2)
    criterion = nn.CrossEntropyLoss()
    statistics = {}
    result = train_epoch(
        OverflowOnlyInAutocast(),
        loader,
        criterion,
        torch.device("cpu"),
        amp_enabled=True,
        amp_statistics=statistics,
    )
    assert result == pytest.approx(float(criterion(values, targets)))
    assert statistics["fp32_fallback_batches"] == 1


def test_nonfinite_fp32_loss_still_stops_training():
    values = torch.tensor([[float("inf"), 1.0]])
    loader = DataLoader(TensorDataset(values, torch.tensor([0])), batch_size=1)
    with pytest.raises(FloatingPointError, match="FP32"):
        train_epoch(
            nn.Identity(), loader, nn.CrossEntropyLoss(), torch.device("cpu"), amp_enabled=True
        )
