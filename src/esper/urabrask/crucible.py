"""Urabrask Crucible v0 (prototype).

Minimal, deterministic hazard evaluator that produces a canonical Leyline BSDS
without executing kernels. Intended as an initial producer before full hazard
battery lands.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import contextlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Mapping, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from esper.karn import BlueprintDescriptor, BlueprintTier
from esper.leyline import leyline_pb2
from esper.core import EsperSettings


@dataclass(slots=True)
class CrucibleConfig:
    high_threshold: float = 0.60
    critical_threshold: float = 0.85


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def run_crucible(
    descriptor: BlueprintDescriptor,
    *,
    artifact_path: Path | None = None,  # reserved for future, not used in v0
    hints: Mapping[str, object] | None = None,
    config: CrucibleConfig | None = None,
) -> leyline_pb2.BSDS:
    cfg = config or CrucibleConfig()
    risk = float(getattr(descriptor, "risk", 0.0) or 0.0)
    tier = getattr(descriptor, "tier", BlueprintTier.BLUEPRINT_TIER_UNSPECIFIED)
    quarantine_only = bool(getattr(descriptor, "quarantine_only", False))

    # Priors: small bump for experimental/high risk tiers; aggressive for quarantine_only
    if tier == BlueprintTier.BLUEPRINT_TIER_EXPERIMENTAL:
        risk += 0.05
    if tier == BlueprintTier.BLUEPRINT_TIER_HIGH_RISK:
        risk += 0.10
    if quarantine_only:
        risk += 0.10
    risk = _clamp01(risk)

    if risk >= cfg.critical_threshold:
        hazard = leyline_pb2.HAZARD_BAND_CRITICAL
    elif risk >= cfg.high_threshold:
        hazard = leyline_pb2.HAZARD_BAND_HIGH
    elif risk >= 0.30:
        hazard = leyline_pb2.HAZARD_BAND_MEDIUM
    else:
        hazard = leyline_pb2.HAZARD_BAND_LOW

    if quarantine_only or hazard == leyline_pb2.HAZARD_BAND_CRITICAL:
        handling = leyline_pb2.HANDLING_CLASS_QUARANTINE
    elif hazard == leyline_pb2.HAZARD_BAND_HIGH:
        handling = leyline_pb2.HANDLING_CLASS_RESTRICTED
    else:
        handling = leyline_pb2.HANDLING_CLASS_STANDARD
    # Ensure quarantine is reflected as at least HIGH hazard in v0
    if quarantine_only and hazard < leyline_pb2.HAZARD_BAND_HIGH:
        hazard = leyline_pb2.HAZARD_BAND_HIGH

    resource_profile = leyline_pb2.RESOURCE_PROFILE_MIXED
    raw_profile = str((hints or {}).get("resource_profile", "") or "").lower()
    if raw_profile in {"gpu", "cuda"}:
        resource_profile = leyline_pb2.RESOURCE_PROFILE_GPU
    elif raw_profile == "cpu":
        resource_profile = leyline_pb2.RESOURCE_PROFILE_CPU

    bsds = leyline_pb2.BSDS(
        version=1,
        blueprint_id=getattr(descriptor, "blueprint_id", ""),
        risk_score=risk,
        hazard_band=hazard,
        handling_class=handling,
        resource_profile=resource_profile,
        recommendation=(
            "Prefer optimizer downgrade; avoid aggressive grafting"
            if hazard in (leyline_pb2.HAZARD_BAND_HIGH, leyline_pb2.HAZARD_BAND_CRITICAL)
            else ""
        ),
        provenance=leyline_pb2.PROVENANCE_URABRASK,
    )
    bsds.issued_at.FromDatetime(datetime.now(tz=UTC))
    return bsds


__all__ = ["CrucibleConfig", "run_crucible"]


# -----------------------------
# Crucible v1 (hazard battery)
# -----------------------------

@dataclass(slots=True)
class CrucibleConfigV1:
    high_threshold: float = 0.60
    critical_threshold: float = 0.85
    grad_explode_threshold: float = 1e3
    grad_vanish_threshold: float = 1e-8
    precision_relerr_threshold: float = 0.10
    oscillation_std_threshold: float = 0.05
    enable_nan_probe: bool = False
    # Internal knobs
    grad_weight_scale: float = 10.0  # larger -> more likely explode
    train_steps: int = 5
    # WP8.2 additions
    memory_watermark_mb_threshold: float = 64.0
    enable_oom_probe: bool = False
    oom_allocation_mb: int = 2048
    simulate_oom: bool = False


def _set_determinism() -> None:
    try:
        torch.manual_seed(123)
        torch.use_deterministic_algorithms(True)  # type: ignore[attr-defined]
    except Exception:
        pass


def _build_tiny_mlp(in_dim: int = 8, hidden: int = 16, out_dim: int = 4, scale: float = 1.0) -> nn.Module:
    model = nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.Tanh(),
        nn.Linear(hidden, out_dim),
    )
    with torch.no_grad():
        for m in model:
            if isinstance(m, nn.Linear):
                m.weight.mul_(scale)
                if m.bias is not None:
                    m.bias.zero_()
    return model


def _hazards_map(**kwargs: str) -> dict[str, str]:
    return {k: v for k, v in kwargs.items()}


def run_crucible_v1(
    descriptor: BlueprintDescriptor,
    *,
    artifact_path: Path | None = None,
    hints: Mapping[str, object] | None = None,
    config: CrucibleConfigV1 | None = None,
) -> Tuple[leyline_pb2.BSDS, dict[str, str]]:
    cfg = config or CrucibleConfigV1()
    # Fold in environment-driven safety flags
    try:
        settings = EsperSettings()
        if settings.urabrask_crucible_memory_watermark_mb is not None:
            cfg.memory_watermark_mb_threshold = float(settings.urabrask_crucible_memory_watermark_mb)
        cfg.enable_oom_probe = bool(settings.urabrask_crucible_allow_oom)
        cfg.simulate_oom = bool(settings.urabrask_crucible_simulate_oom)
    except Exception:
        pass
    _t0 = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None  # type: ignore[attr-defined]
    _t1 = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None  # type: ignore[attr-defined]
    if _t0 is not None and _t1 is not None:
        try:
            _t0.record()
        except Exception:
            _t0 = _t1 = None
    # Note: avoid setting global deterministic algorithms to not affect CUDA tests elsewhere

    # Inputs
    risk_base = float(getattr(descriptor, "risk", 0.0) or 0.0)
    quarantine_only = bool(getattr(descriptor, "quarantine_only", False))

    # 1) Gradient instability
    x = torch.randn(4, 8)
    y = torch.randn(4, 4)
    model = _build_tiny_mlp(scale=cfg.grad_weight_scale)
    optim = torch.optim.SGD(model.parameters(), lr=0.001)
    grad_norms = []
    for _ in range(3):
        optim.zero_grad(set_to_none=True)
        out = model(x)
        loss = F.mse_loss(out, y)
        loss.backward()
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += float(p.grad.detach().norm().item())
        grad_norms.append(total_norm)
        optim.step()
    grad_max = max(grad_norms) if grad_norms else 0.0
    grad_min = min(grad_norms) if grad_norms else 0.0
    grad_status = "ok"
    if grad_max > cfg.grad_explode_threshold:
        grad_status = "explode"
    elif grad_min < cfg.grad_vanish_threshold:
        grad_status = "vanish"

    # 2) NaN/Inf probe (optional)
    nan_status = "ok"
    if cfg.enable_nan_probe:
        a = torch.tensor([-1.0, 0.0, 1.0])
        b = torch.log(a)  # nan for negative
        if torch.isnan(b).any() or torch.isinf(b).any():
            nan_status = "present"

    # 3) Precision sensitivity (quantization emulation)
    with torch.inference_mode():
        base = model(x).abs().mean().item()
        q = torch.round(model(x) * 10.0) / 10.0  # emulate lower precision
        q_mean = q.abs().mean().item()
        rel_err = 0.0
        if base > 0:
            rel_err = abs(q_mean - base) / base
    prec_status = "sensitive" if rel_err > cfg.precision_relerr_threshold else "ok"

    # 4) Oscillation (tiny train steps)
    x2 = torch.randn(8, 8)
    y2 = torch.randn(8, 4)
    model2 = _build_tiny_mlp(scale=1.0)
    opt2 = torch.optim.SGD(model2.parameters(), lr=0.01)
    losses = []
    for _ in range(max(1, cfg.train_steps)):
        opt2.zero_grad(set_to_none=True)
        out2 = model2(x2)
        loss2 = F.mse_loss(out2, y2)
        losses.append(float(loss2.item()))
        loss2.backward()
        opt2.step()
    import statistics

    osc_std = statistics.pstdev(losses) if len(losses) > 1 else 0.0
    osc_status = "high" if osc_std > cfg.oscillation_std_threshold else "ok"

    # 5) Memory watermark (process RSS delta)
    mem_status = "ok"
    try:
        import psutil  # type: ignore

        proc = psutil.Process()
        rss_before = float(proc.memory_info().rss) / (1024.0 * 1024.0)
        # Light allocation workload to perturb memory slightly
        _tmp = [torch.randn(16, 16) for _ in range(2)]
        rss_after = float(proc.memory_info().rss) / (1024.0 * 1024.0)
        delta = max(0.0, rss_after - rss_before)
        if delta > max(0.0, float(cfg.memory_watermark_mb_threshold)):
            mem_status = "high"
    except Exception:
        mem_status = "ok"

    # 6) OOM risk probe (guarded)
    oom_status = "ok"
    if cfg.enable_oom_probe:
        try:
            if cfg.simulate_oom:
                raise RuntimeError("simulated_oom")
            # Attempt to allocate a large buffer on CPU; immediately free
            elements = max(1, int((cfg.oom_allocation_mb * 1024 * 1024) / 4))
            t = torch.empty(elements, dtype=torch.float32)
            del t
        except Exception:
            oom_status = "risk"

    # Aggregate scoring
    score = 0.0
    if nan_status == "present":
        score += 0.6
    if grad_status == "explode":
        score += 0.3
    elif grad_status == "vanish":
        score += 0.2
    if prec_status == "sensitive":
        score += 0.2
    if osc_status == "high":
        score += 0.2
    if mem_status == "high":
        score += 0.2
    if oom_status == "risk":
        score += 0.3
    score = min(1.0, risk_base + score)

    # Map to band and handling
    if score >= cfg.critical_threshold or quarantine_only:
        hazard = leyline_pb2.HAZARD_BAND_CRITICAL
    elif score >= cfg.high_threshold:
        hazard = leyline_pb2.HAZARD_BAND_HIGH
    elif score >= 0.30:
        hazard = leyline_pb2.HAZARD_BAND_MEDIUM
    else:
        hazard = leyline_pb2.HAZARD_BAND_LOW

    if quarantine_only or hazard == leyline_pb2.HAZARD_BAND_CRITICAL:
        handling = leyline_pb2.HANDLING_CLASS_QUARANTINE
    elif hazard == leyline_pb2.HAZARD_BAND_HIGH:
        handling = leyline_pb2.HANDLING_CLASS_RESTRICTED
    else:
        handling = leyline_pb2.HANDLING_CLASS_STANDARD

    resource_profile = leyline_pb2.RESOURCE_PROFILE_MIXED
    raw_profile = str((hints or {}).get("resource_profile", "") or "").lower()
    if raw_profile in {"gpu", "cuda"}:
        resource_profile = leyline_pb2.RESOURCE_PROFILE_GPU
    elif raw_profile == "cpu":
        resource_profile = leyline_pb2.RESOURCE_PROFILE_CPU

    bsds = leyline_pb2.BSDS(
        version=1,
        blueprint_id=getattr(descriptor, "blueprint_id", ""),
        risk_score=float(score),
        hazard_band=hazard,
        handling_class=handling,
        resource_profile=resource_profile,
        recommendation=(
            "Prefer optimizer downgrade; avoid aggressive grafting"
            if hazard in (leyline_pb2.HAZARD_BAND_HIGH, leyline_pb2.HAZARD_BAND_CRITICAL)
            else ""
        ),
        provenance=leyline_pb2.PROVENANCE_URABRASK,
    )
    bsds.issued_at.FromDatetime(datetime.now(tz=UTC))
    hazards = _hazards_map(
        grad_instability=grad_status,
        nan_inf=nan_status,
        precision="sensitive" if prec_status == "sensitive" else "ok",
        oscillation=osc_status,
        memory_watermark=mem_status,
        oom_risk=oom_status,
    )
    # Measure duration
    duration_ms = 0.0
    if _t0 is not None and _t1 is not None:
        try:
            _t1.record()
            torch.cuda.synchronize()  # type: ignore[attr-defined]
            duration_ms = float(_t0.elapsed_time(_t1))
        except Exception:
            duration_ms = 0.0
    else:
        # Fallback: coarse wall time not tracked; leave 0.0 to keep overhead tiny
        duration_ms = 0.0

    # Persist minimal result bundle per WP8.1
    try:
        settings = EsperSettings()
        _persist_result_bundle(
            blueprint_id=bsds.blueprint_id,
            bsds=bsds,
            hazards=hazards,
            duration_ms=duration_ms,
            config_snapshot=asdict(cfg),
            settings=settings,
        )
    except Exception:
        pass

    return bsds, hazards


def _persist_result_bundle(
    *,
    blueprint_id: str,
    bsds: leyline_pb2.BSDS,
    hazards: Mapping[str, str],
    duration_ms: float,
    config_snapshot: Mapping[str, object],
    settings: EsperSettings,
) -> None:
    # Build path
    artifacts_root = Path(settings.urabrask_crucible_artifacts_dir)
    bundle_dir = artifacts_root / blueprint_id
    bundle_dir.mkdir(parents=True, exist_ok=True)
    # File name with issued_at timestamp to ensure stable order
    issued = bsds.issued_at.ToDatetime().replace(tzinfo=UTC).isoformat().replace("+00:00", "Z")
    safe_issued = issued.replace(":", "-")
    file_path = bundle_dir / f"{safe_issued}.json"
    # Prepare minimal JSON bundle
    bsds_json = {
        "version": int(bsds.version),
        "blueprint_id": bsds.blueprint_id,
        "risk_score": float(bsds.risk_score),
        "hazard_band": leyline_pb2.HazardBand.Name(bsds.hazard_band).replace("HAZARD_BAND_", ""),
        "handling_class": leyline_pb2.HandlingClass.Name(bsds.handling_class).replace("HANDLING_CLASS_", "").lower(),
        "resource_profile": leyline_pb2.ResourceProfile.Name(bsds.resource_profile).replace("RESOURCE_PROFILE_", "").lower(),
        "provenance": leyline_pb2.Provenance.Name(bsds.provenance).replace("PROVENANCE_", ""),
        "issued_at": issued,
    }
    payload = {
        "artifact_version": 1,
        "crucible_version": "v1",
        "blueprint_id": blueprint_id,
        "bsds": bsds_json,
        "hazards": dict(hazards),
        "timings": {"duration_ms": float(duration_ms)},
        "config": dict(config_snapshot),
    }
    try:
        import json

        with file_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, sort_keys=True, separators=(",", ":"))
            f.write("\n")
    except Exception:
        return
    # Retention by count (keep newest N)
    try:
        keep = max(1, int(settings.urabrask_crucible_artifacts_keep))
    except Exception:
        keep = 5
    try:
        entries = sorted([p for p in bundle_dir.glob("*.json") if p.is_file()])
        excess = max(0, len(entries) - keep)
        for p in entries[:excess]:
            with contextlib.suppress(Exception):
                p.unlink()
    except Exception:
        pass
    
