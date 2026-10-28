"""Measure batch-8 AMP steps at production resolution; software evidence only."""

import gc
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from pulmonary.common import read_config, save_json, seed_everything
from pulmonary.image_models import configure_stage, train_epoch
from pulmonary.networks import ImageEncoder
from pulmonary.precision import grad_scaler, use_amp, validate_training_limits


def main():
    config = read_config()
    validate_training_limits(config)
    if not torch.cuda.is_available():
        raise RuntimeError("This check requires the NVIDIA GPU")
    seed_everything(config["seed"])
    device = torch.device("cuda")
    enabled = use_amp(config, device)
    inputs = torch.randn(8, 3, config["image_size"], config["image_size"])
    targets = torch.tensor([0, 1] * 4)
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=8, shuffle=False)
    report = {
        "status": "running",
        "gpu": torch.cuda.get_device_name(),
        "batch_size": 8,
        "image_size": config["image_size"],
        "amp_enabled": enabled,
        "clinical_evidence": False,
        "models": {},
    }
    output = Path("reports/generated/gpu_budget_check.json")
    for name in config["image_models"]:
        gc.collect()
        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info()
        torch.cuda.reset_peak_memory_stats()
        model = ImageEncoder(name, pretrained=True, dropout=config["dropout"]).to(device)
        observed_dtypes = []
        hook = model.head[-1].register_forward_hook(
            lambda _m, _i, out, values=observed_dtypes: values.append(str(out.dtype))
        )
        scaler = grad_scaler(enabled)
        criterion = nn.CrossEntropyLoss().to(device)
        before = model.head[-1].weight.detach().cpu().clone()
        losses = {}
        for phase in ["head", "finetune"]:
            configure_stage(model, phase)
            optimizer = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad], lr=config["learning_rate"]
            )
            losses[phase] = train_epoch(
                model, loader, criterion, device, optimizer, scaler=scaler, amp_enabled=enabled
            )
            del optimizer
        torch.cuda.synchronize()
        if observed_dtypes != ["torch.float16", "torch.float16"]:
            raise AssertionError(f"AMP did not produce FP16 classifier outputs: {observed_dtypes}")
        if torch.equal(before, model.head[-1].weight.detach().cpu()):
            raise AssertionError("Scaled optimization did not update the head")
        report["models"][name] = {
            "losses": losses,
            "output_dtypes": observed_dtypes,
            "scaler_enabled": scaler.is_enabled(),
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
            "peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
            "free_before_mib": free / 1024**2,
            "total_device_mib": total / 1024**2,
        }
        save_json(output, report)
        print(name, report["models"][name], flush=True)
        hook.remove()
        del model, scaler
    report["status"] = "passed"
    save_json(output, report)
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
