---
title: PROTOTYPE CODE – FULL-FIDELITY MANAGED GERMINATION
split_mode: consolidated
appendix: "A"
coauthors:
  - John Morrissey
  - Codex CLI (OpenAI)
generated_by: scripts/split_paper.py
---

# Appendix A: Prototype Code – Full-Fidelity Managed Germination
import os
import random
import threading
import time
from collections import deque
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

################################################################################

# 1. CORE INFRASTRUCTURE #

################################################################################

class SeedManager:
    """Central registry for SentinelSeed instances.

    Thread‑safe singleton that tracks seed state, telemetry, and germination
    lineage. It also exposes an atomic germination request that can be called
    concurrently from controller logic.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance.seeds = {}
                cls._instance.germination_log = []
        return cls._instance

    # ---------------------------------------------------------------------
    #  Registry helpers
    # ---------------------------------------------------------------------

    def register_seed(self, seed_module: "SentinelSeed", seed_id: str) -> None:
        self.seeds[seed_id] = {
            "module": seed_module,
            "buffer": deque(maxlen=500),  # activation buffer for stats
            "status": "dormant",         # dormant | active | failed_germination
            "telemetry": {"interface_drift": 0.0},
        }
        print(f"SeedManager ▶ Registered '{seed_id}'.")

    def get_seed_info(self, seed_id: str):
        return self.seeds.get(seed_id)

    # ---------------------------------------------------------------------
    #  Germination
    # ---------------------------------------------------------------------

    def request_germination(
        self,
        seed_id: str,
        step: int,
        init_type: str = "zero_init",
        gm_path: str | None = None,
    ) -> bool:
        """Attempt to activate a dormant seed.

        Returns True on success so the caller can refresh the optimiser.
        """
        with self._lock:
            info = self.get_seed_info(seed_id)
            if not info or info["status"] != "dormant":
                return False

            # Simulate hardware failure 15 % of the time.
            if random.random() < 0.15:
                print(f"\N{RED CIRCLE}  SeedManager ▶ Simulated GERMINATION FAILURE for '{seed_id}'.")
                info["status"] = "failed_germination"
                self._log_event(step, seed_id, "failure", "simulated hardware error")
                return False

            print(
                f"\N{LARGE GREEN CIRCLE}  SeedManager ▶ Germinating '{seed_id}' "
                f"using {init_type} ..."
            )

            ok = info["module"].germinate(init_type=init_type, gm_path=gm_path)
            if ok:
                info["status"] = "active"
                self._log_event(step, seed_id, "success", init_type)
            return ok

    # ------------------------------------------------------------------
    #  Telemetry
    # ------------------------------------------------------------------

    def _log_event(self, step: int, seed_id: str, status: str, details: str) -> None:
        self.germination_log.append(
            {
                "step": step,
                "timestamp": time.time(),
                "seed_id": seed_id,
                "status": status,
                "details": details,
            }
        )

    def print_audit_log(self) -> None:
        print("\n── Germination Audit Log ──────────────")
        if not self.germination_log:
            print("<no events>")
        else:
            for e in self.germination_log:
                print(
                    f"step={e['step']:<4} | seed={e['seed_id']:<15} | "
                    f"status={e['status']:<6} | details={e['details']}"
                )
        print("──────────────────────────────────────\n")

################################################################################

# 2. CONTROLLER #

################################################################################

class KasminaMicro:
    """Very simple plateau‑trigger controller.

    In production this would be replaced by a RL or heuristic policy.
    """

    def __init__(self, manager: SeedManager, patience: int = 20, delta: float = 1e-4):
        self.mgr = manager
        self.patience = patience
        self.delta = delta
        self.plateau = 0
        self.prev_loss = float("inf")
        print(
            "Kasmina ▶ initialised with patience="
            f"{self.patience} and Δ={self.delta}."
        )

    # ------------------------------------------------------------------
    #  Decide if we should invoke a seed and optionally return a flag so
    #  the caller can rebuild the optimiser.
    # ------------------------------------------------------------------

    def step(self, step_idx: int, val_loss: float) -> bool:
        rebuild = False
        if abs(val_loss - self.prev_loss) < self.delta:
            self.plateau += 1
        else:
            self.plateau = 0
        self.prev_loss = val_loss

        if self.plateau < self.patience:
            return rebuild

        self.plateau = 0  # reset
        candidate = self._select_seed()
        if not candidate:
            return rebuild

        init_type = "Germinal Module (GM)" if random.random() > 0.5 else "zero_init"
        ok = self.mgr.request_germination(candidate, step_idx, init_type, gm_path="gm.pth")
        return ok  # if True, caller should rebuild optimiser

    # ------------------------------------------------------------------
    #  Helper
    # ------------------------------------------------------------------

    def _select_seed(self):
        dormant = {
            sid: info for sid, info in self.mgr.seeds.items() if info["status"] == "dormant"
        }
        if not dormant:
            return None
        # Choose the seed with *lowest* variance (most starving).
        scores = {
            sid: info["module"].get_health_signal() for sid, info in dormant.items()
        }
        return min(scores, key=scores.get)

################################################################################

# 3. MODEL COMPONENTS #

################################################################################

class SentinelSeed(nn.Module):
    """Drop‑in residual block — dormant until germinated."""

    def __init__(self, seed_id: str, dim: int = 32):
        super().__init__()
        self.seed_id = seed_id
        self.mgr = SeedManager()
        self.mgr.register_seed(self, seed_id)

        self.child = nn.Sequential(
            nn.Linear(dim, 16),
            nn.ReLU(),
            nn.Linear(16, dim),
        )
        self._zero_init(self.child)
        self.set_trainable(False)

    # ------------------------------- lifecycle ---------------------------------

    def germinate(self, init_type: str = "zero_init", gm_path: str | None = None) -> bool:
        try:
            if init_type == "Germinal Module (GM)" and gm_path and os.path.exists(gm_path):
                self.child.load_state_dict(torch.load(gm_path))
                print(f"Seed '{self.seed_id}' ▶ GM loaded from '{gm_path}'.")
            else:
                self._kaiming_init(self.child)
            self.set_trainable(True)
            return True
        except Exception as exc:  # pragma: no cover
            print(f"\N{RED CIRCLE}  '{self.seed_id}' ▶ germination failed: {exc}")
            return False

    # ----------------------------- forward pass --------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        info = self.mgr.get_seed_info(self.seed_id)
        status = info["status"]
        if status != "active":
            if status == "dormant":
                info["buffer"].append(x.detach())  # collect stats
            return x  # identity

        residual = self.child(x)
        out = x + residual
        drift = 1.0 - F.cosine_similarity(x, out, dim=-1).mean().item()
        info["telemetry"]["interface_drift"] = drift
        return out

    # ----------------------------- diagnostics ---------------------------------

    def get_health_signal(self) -> float:
        buf = self.mgr.get_seed_info(self.seed_id)["buffer"]
        if len(buf) < 20:
            return 1.0  # optimistic until we have data
        variance = torch.var(torch.stack(list(buf))).item()
        return max(variance, 1e-6)

    # --------------------------- utils / helpers ------------------------------

    @staticmethod
    def _zero_init(module: nn.Module) -> None:
        for m in module.modules():
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @staticmethod
    def _kaiming_init(module: nn.Module) -> None:
        for m in module.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def set_trainable(self, flag: bool) -> None:
        for p in self.parameters():
            p.requires_grad = flag

class BaseNet(nn.Module):
    """Frozen backbone with two insertion points."""

    def __init__(self, seed_a: SentinelSeed, seed_b: SentinelSeed):
        super().__init__()
        self.fc1 = nn.Linear(2, 32)
        self.seed_a = seed_a
        self.fc2 = nn.Linear(32, 32)
        self.seed_b = seed_b
        self.out = nn.Linear(32, 2)

        self._freeze_except_seeds()

    # ------------------------------------------------------------------
    #  Helpers
    # ------------------------------------------------------------------

    def _freeze_except_seeds(self):
        for m in self.modules():
            trainable = isinstance(m, SentinelSeed)
            for p in m.parameters(recurse=False):
                p.requires_grad = trainable

    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor):  # type: ignore[override]
        x = F.relu(self.fc1(x))
        x = self.seed_a(x)
        x = F.relu(self.fc2(x))
        x = self.seed_b(x)
        return self.out(x)

################################################################################

# 4. TRAINING LOOP #

################################################################################

def create_dummy_gm(path: str = "gm.pth", dim: int = 32) -> None:
    """Persist an untrained module so the GM code path has something to load."""
    if os.path.exists(path):
        return
    print("Creating placeholder Germinal Module …")
    tmp = nn.Sequential(nn.Linear(dim, 16), nn.ReLU(), nn.Linear(16, dim))
    torch.save(tmp.state_dict(), path)

def train_demo(n_steps: int = 800):  # pragma: no cover
    # ------------------------------------------------------------------
    #  Dataset
    # ------------------------------------------------------------------
    X, y = make_moons(1000, noise=0.2, random_state=42)
    X = StandardScaler().fit_transform(X).astype("float32")
    y = y.astype("int64")
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    X_tr, X_val = map(torch.from_numpy, (X_tr, X_val))
    y_tr, y_val = map(torch.from_numpy, (y_tr, y_val))

    # ------------------------------------------------------------------
    #  Model + manager
    # ------------------------------------------------------------------
    mgr = SeedManager()
    seed1 = SentinelSeed("bottleneck_1")
    seed2 = SentinelSeed("bottleneck_2")
    model = BaseNet(seed1, seed2)
    ctrl = KasminaMicro(mgr)

    # ------------------------------------------------------------------
    #  Stage 1: warm‑up backbone only
    # ------------------------------------------------------------------
    create_dummy_gm()
    for m in model.modules():
        if isinstance(m, SentinelSeed):
            m.set_trainable(False)
        else:
            for p in m.parameters(recurse=False):
                p.requires_grad = True
    warm_opt = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    model.train()
    for _ in range(300):
        warm_opt.zero_grad()
        loss = F.cross_entropy(model(X_tr), y_tr)
        loss.backward()
        warm_opt.step()
    print("Backbone pre‑trained, freezing …")

    # freeze backbone, leave seeds dormant (not trainable until active)
    model._freeze_except_seeds()

    # ------------------------------------------------------------------
    #  Stage 2: main loop
    # ------------------------------------------------------------------
    def build_opt() -> optim.Optimizer:
        return optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)

    opt = build_opt()
    prev_val = float("inf")
    for step in range(n_steps):
        model.train()
        opt.zero_grad()
        F.cross_entropy(model(X_tr), y_tr).backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            val_loss = F.cross_entropy(model(X_val), y_val).item()

        if ctrl.step(step, val_loss):  # seed activated ⇒ refresh optimiser
            opt = build_opt()

        if step % 100 == 0:
            acc = (model(X_val).argmax(1) == y_val).float().mean().item()
            print(
                f"step={step:>3} | val_loss={val_loss:6.4f} | val_acc={acc:.2%}"
            )
            for sid, info in mgr.seeds.items():
                print(
                    f"   ↳ {sid:<13} status={info['status']:<10} "
                    f"var={info['module'].get_health_signal():.4f} "
                    f"drift={info['telemetry']['interface_drift']:.4f}"
                )
        prev_val = val_loss

    mgr.print_audit_log()

################################################################################

# 5. ENTRY‑POINT #

################################################################################

if __name__ == "__main__":
    train_demo()
