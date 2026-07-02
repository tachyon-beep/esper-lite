"""GATE 2 learnability probe — reward-credit term (PDR-0010, esper-lite-254175df90).

Measures whether a once-per-episode sparse credit attributed to a FOSSILIZE
action can propagate into this recurrent PPO's policy-gradient signal, per the
drl-expert three-tier design (2026-07-02):

  Tier 0b/1 (every batch, both arms, analytic-empirical): paired GAE recompute
    on the live rollout buffer — inject a probe credit, recompute advantages
    through the production pipeline (RewardNormalizer scale -> GAE ->
    denormalized values), and record the realized Δadv at fossilize steps for
    BOTH routings (retro-write at t_f, terminal-flush at the last step),
    against the advantage-noise floor σ_A. GAE is linear in rewards, so the
    per-unit-δ effect is exact; a self-check asserts the linear identity.

  Tier 2 (paired arms, same seed): the `control` arm restores the buffer after
    measuring; `retro`/`terminal` arms keep their injection so agent.update()
    trains on it. Behavioral endpoint: P(op=FOSSILIZE | fossilize-legal),
    read via a no-grad forward over the buffer states conditioned on the
    stored per-step LSTM hiddens (the TELEMETRY-ONLY pre_step_hidden buffer).

The probe wraps PPOCoordinator.run_update at class level; production code is
untouched. Rollback-contaminated envs are excluded from injection and
measurement (their reward prefix is forfeited by handle_rollbacks).

Usage:
  uv run python scripts/gate2_learnability_probe.py \
      --arm retro --delta-raw 2.0 --seed 41 --rounds 20 --envs 12 \
      --episode-length 150 --device cuda:0 --gpu-preload \
      --out telemetry/gate2_probe/retro_s41.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from esper.leyline.factored_actions import LifecycleOp
from esper.simic.training.config import TrainingConfig
from esper.simic.training.ppo_coordinator import PPOCoordinator
from esper.simic.training.vectorized import train_ppo_vectorized

FOSS = int(LifecycleOp.FOSSILIZE)

# The n=5 J-read regime (PDR-0009): the probe MUST train in the same regime the
# gate generalizes to — auto-forward gates, entropy anneal 0.15->0.08 with
# per-head multipliers, gae_lambda=0.95, value_coef=0.25, AMP. Loaded exactly
# like scripts/causal_contribution_r1_pilot.py loads it.
DEFAULT_CONFIG = "configs/config-3slot-3seed-suppress-slot-r0c0.json"
_HARNESS_KEY = "_causal_contribution_r1"


class Gate2Probe:
    """Per-batch injection + measurement, installed around PPOCoordinator.run_update."""

    def __init__(
        self,
        arm: str,
        delta_raw: float,
        out_path: Path,
        self_check: bool,
    ) -> None:
        if arm not in ("control", "retro", "terminal"):
            raise ValueError(f"Unknown arm: {arm}")
        self.arm = arm
        self.delta_raw = delta_raw
        self.out_path = out_path
        self.self_check = self_check
        self.batch_idx = 0
        self._orig_run_update = None
        self._pending: dict | None = None
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Truncate any prior probe output for this run identity.
        out_path.write_text("")

    # ------------------------------------------------------------------ install

    def install(self) -> None:
        self._orig_run_update = PPOCoordinator.run_update
        probe = self
        orig = self._orig_run_update

        def wrapped(coord: PPOCoordinator, *args, **kwargs):
            probe._pre_update(coord)
            metrics, update_skipped, ppo_time = orig(coord, *args, **kwargs)
            probe._post_update(metrics, update_skipped)
            return metrics, update_skipped, ppo_time

        PPOCoordinator.run_update = wrapped

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _gae(agent) -> None:
        """Recompute GAE exactly as agent.update() does (non-HRA leg)."""
        agent.buffer.compute_advantages_and_returns(
            gamma=agent.gamma,
            gae_lambda=agent.gae_lambda,
            value_normalizer=agent.value_normalizer,
        )

    @staticmethod
    def _tensor_stats(t: torch.Tensor) -> dict:
        if t.numel() == 0:
            return {"n": 0, "mean": None, "median": None, "std": None}
        return {
            "n": int(t.numel()),
            "mean": float(t.mean()),
            "median": float(t.median()),
            "std": float(t.std()) if t.numel() > 1 else 0.0,
        }

    def _propensity(self, agent, buffer, valid: torch.Tensor) -> dict:
        """P(op=FOSSILIZE | fossilize-legal) via a no-grad forward over buffer
        states conditioned on the stored per-step LSTM hiddens."""
        idx = torch.nonzero(valid)
        if idx.numel() == 0:
            return {"eligible": {"n": 0}, "op_entropy_eligible_mean": None}
        rows, cols = idx[:, 0], idx[:, 1]
        states = buffer.states[rows, cols].unsqueeze(1)  # [K, 1, state_dim]
        bps = buffer.blueprint_indices[rows, cols].unsqueeze(1)  # [K, 1, num_slots]
        op_mask = buffer.op_masks[rows, cols].unsqueeze(1)  # [K, 1, num_ops]
        h = buffer.hidden_h[rows, cols].permute(1, 0, 2).contiguous()  # [layers, K, hid]
        c = buffer.hidden_c[rows, cols].permute(1, 0, 2).contiguous()
        device_type = states.device.type
        # Mirrors the Q(s, op) telemetry forward in ppo_agent.py: pure-FP32,
        # no_grad, autocast disabled so the training-dtype cast cache is untouched.
        with torch.autocast(device_type=device_type, enabled=False), torch.no_grad():
            out = agent.policy.network.forward(
                state=states,
                blueprint_indices=bps,
                hidden=(h, c),
                op_mask=op_mask,
            )
        probs = F.softmax(out["op_logits"].float(), dim=-1)[:, 0, :]  # [K, num_ops]
        mask_flat = op_mask[:, 0, :]
        eligible = mask_flat[:, FOSS]
        p_foss = probs[eligible, FOSS]
        # Conditional (legal-renormalized) op entropy on eligible steps.
        probs_e = probs[eligible]
        mask_e = mask_flat[eligible]
        p_masked = torch.where(mask_e, probs_e, torch.zeros_like(probs_e))
        p_masked = p_masked / p_masked.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        ent = -(p_masked * (p_masked.clamp_min(1e-12)).log()).sum(dim=-1)
        return {
            "eligible": self._tensor_stats(p_foss),
            "op_entropy_eligible_mean": float(ent.mean()) if ent.numel() else None,
        }

    # ------------------------------------------------------------------ hooks

    def _pre_update(self, coord: PPOCoordinator) -> None:
        agent = coord.agent
        buffer = agent.buffer
        self.batch_idx += 1
        if len(buffer) == 0:
            self._pending = {"batch": self.batch_idx, "skipped": "empty_buffer"}
            return
        if agent.hra_value_decomposition:
            raise RuntimeError("GATE 2 probe assumes the non-HRA (single-stream) leg.")

        device = buffer.rewards.device
        n_envs, max_steps = buffer.rewards.shape
        step_counts = torch.tensor(buffer.step_counts, device=device, dtype=torch.long)
        valid = (
            torch.arange(max_steps, device=device)[None, :] < step_counts[:, None]
        )

        # Rollback-contaminated envs: prefix rewards were forfeited by
        # handle_rollbacks; exclude them from injection AND measurement.
        rb_env = (buffer.rollback_severity != 0).any(dim=1) | (
            buffer.rollback_transition_types != 0
        ).any(dim=1)
        clean = valid & ~rb_env[:, None]

        foss = (
            (buffer.effective_op_actions == FOSS)
            & clean
            & ~buffer.forced_actions
        )
        n_foss = int(foss.sum())
        gamma_lambda = agent.gamma * agent.gae_lambda
        delta_buf = coord.reward_normalizer.divide_by_std(self.delta_raw)

        # Lifecycle diagnostics: per-op legality (steps where each op was legal)
        # and executed-op histogram — locates where the lifecycle stalls when
        # fossilize-eligibility is zero.
        op_legal_counts = {
            op.name: int((buffer.op_masks[:, :, int(op)] & clean).sum())
            for op in LifecycleOp
        }
        executed, counts = torch.unique(
            buffer.effective_op_actions[clean], return_counts=True
        )
        op_executed_counts = {
            LifecycleOp(int(o)).name: int(c) for o, c in zip(executed, counts)
        }

        record: dict = {
            "batch": self.batch_idx,
            "arm": self.arm,
            "delta_raw": self.delta_raw,
            "delta_buf": delta_buf,
            "n_valid_steps": int(valid.sum()),
            "n_rollback_envs": int(rb_env.sum()),
            "n_foss_steps": n_foss,
            "n_foss_forced": int(
                ((buffer.effective_op_actions == FOSS) & clean & buffer.forced_actions).sum()
            ),
            "op_legal_counts": op_legal_counts,
            "op_executed_counts": op_executed_counts,
        }

        rewards_orig = buffer.rewards.clone()
        self._gae(agent)
        adv_base = buffer.advantages.clone()
        record["sigma_A_raw"] = float(adv_base[clean].std())
        record["adv_base_at_foss"] = self._tensor_stats(adv_base[foss])

        if n_foss > 0:
            # Routing (i): retro-write at each fossilize timestep.
            buffer.rewards[foss] += delta_buf
            self._gae(agent)
            dadv_retro = (buffer.advantages - adv_base)[foss]
            record["dadv_retro_at_foss"] = self._tensor_stats(dadv_retro)
            buffer.rewards.copy_(rewards_orig)

            # Routing (ii): terminal-flush on envs that have >=1 fossilize step.
            env_has_foss = foss.any(dim=1)
            term_rows = torch.nonzero(env_has_foss).squeeze(-1)
            last_idx = (step_counts - 1).clamp(min=0)
            buffer.rewards[term_rows, last_idx[term_rows]] += delta_buf
            self._gae(agent)
            dadv_term = (buffer.advantages - adv_base)[foss]
            record["dadv_term_at_foss"] = self._tensor_stats(dadv_term)
            buffer.rewards.copy_(rewards_orig)

            if self.self_check:
                self._assert_linearity(
                    buffer, foss, step_counts, adv_base, delta_buf, gamma_lambda, agent
                )

            # Persistent arm injection (the buffer agent.update() will train on).
            if self.arm == "retro":
                buffer.rewards[foss] += delta_buf
            elif self.arm == "terminal":
                buffer.rewards[term_rows, last_idx[term_rows]] += delta_buf
        else:
            record["dadv_retro_at_foss"] = {"n": 0}
            record["dadv_term_at_foss"] = {"n": 0}

        record["propensity"] = self._propensity(agent, buffer, clean)
        self._pending = record

    def _assert_linearity(
        self, buffer, foss, step_counts, adv_base, delta_buf, gamma_lambda, agent
    ) -> None:
        """GAE linearity identities, exact up to fp tolerance:
        retro: Δadv at each env's LAST fossilize step == delta_buf;
        terminal: Δadv at that step == delta_buf * (γλ)^(T-1 - t_f)."""
        rewards_orig = buffer.rewards.clone()
        for env in torch.nonzero(foss.any(dim=1)).squeeze(-1).tolist():
            t_last = int(torch.nonzero(foss[env]).max())
            t_end = int(step_counts[env]) - 1

            buffer.rewards[env, t_last] += delta_buf
            self._gae(agent)
            got_retro = float(buffer.advantages[env, t_last] - adv_base[env, t_last])
            buffer.rewards.copy_(rewards_orig)

            buffer.rewards[env, t_end] += delta_buf
            self._gae(agent)
            got_term = float(buffer.advantages[env, t_last] - adv_base[env, t_last])
            buffer.rewards.copy_(rewards_orig)

            want_term = delta_buf * gamma_lambda ** (t_end - t_last)
            if not math.isclose(got_retro, delta_buf, rel_tol=1e-3, abs_tol=1e-5):
                raise AssertionError(
                    f"linearity(retro) env={env} t={t_last}: got {got_retro}, want {delta_buf}"
                )
            if not math.isclose(got_term, want_term, rel_tol=1e-3, abs_tol=1e-5):
                raise AssertionError(
                    f"linearity(terminal) env={env} t={t_last}: got {got_term}, want {want_term}"
                )
        self._gae(agent)  # leave buffer advantages consistent with original rewards

    def _post_update(self, metrics: dict, update_skipped: bool) -> None:
        record = self._pending
        self._pending = None
        if record is None:
            return
        record["update_skipped"] = update_skipped
        for key in (
            "policy_loss",
            "value_loss",
            "entropy",
            "approx_kl",
            "explained_variance",
            "pre_norm_adv_mean",
            "pre_norm_adv_std",
            "advantage_std_floored",
        ):
            if key in metrics:
                record[key] = metrics[key]
        with self.out_path.open("a") as f:
            f.write(json.dumps(record) + "\n")


def summarize(out_path: Path) -> dict:
    records = [json.loads(line) for line in out_path.read_text().splitlines() if line]
    batches = [r for r in records if "skipped" not in r]
    with_foss = [r for r in batches if r["n_foss_steps"] > 0]

    def _series(key: str) -> list[float]:
        return [
            r[key]["mean"]
            for r in with_foss
            if r.get(key, {}).get("n", 0) > 0 and r[key]["mean"] is not None
        ]

    def _ratio_series(key: str) -> list[float]:
        return [
            r[key]["mean"] / r["sigma_A_raw"]
            for r in with_foss
            if r.get(key, {}).get("n", 0) > 0
            and r[key]["mean"] is not None
            and r["sigma_A_raw"] > 0
        ]

    summary = {
        "batches": len(batches),
        "batches_with_foss": len(with_foss),
        "total_foss_steps": sum(r["n_foss_steps"] for r in batches),
        "sigma_A_raw_median": (
            statistics.median(r["sigma_A_raw"] for r in batches) if batches else None
        ),
        "dadv_retro_mean": statistics.mean(_series("dadv_retro_at_foss"))
        if _series("dadv_retro_at_foss")
        else None,
        "dadv_term_mean": statistics.mean(_series("dadv_term_at_foss"))
        if _series("dadv_term_at_foss")
        else None,
        "dadv_retro_over_sigmaA_median": (
            statistics.median(_ratio_series("dadv_retro_at_foss"))
            if _ratio_series("dadv_retro_at_foss")
            else None
        ),
        "dadv_term_over_sigmaA_median": (
            statistics.median(_ratio_series("dadv_term_at_foss"))
            if _ratio_series("dadv_term_at_foss")
            else None
        ),
        "p_foss_eligible_series": [
            r["propensity"]["eligible"]["mean"]
            for r in batches
            if r["propensity"]["eligible"].get("n", 0) > 0
        ],
    }
    return summary


def load_regime_kwargs(config_path: str) -> dict:
    """Load the n=5 regime hyperparams exactly as the causal pilot does."""
    raw = json.loads(Path(config_path).read_text())
    raw.pop(_HARNESS_KEY, None)  # driver-only metadata, not TrainingConfig schema
    return TrainingConfig.from_dict(raw).to_train_kwargs()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=("control", "retro", "terminal"))
    parser.add_argument("--delta-raw", type=float, default=2.0,
                        help="Injected credit in RAW reward units (pre-registered "
                        "delta*; scaled into buffer units via divide_by_std)")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--config", default=DEFAULT_CONFIG,
                        help="Regime config JSON (default: the n=5 J-read config)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--gpu-preload", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--self-check", action="store_true",
                        help="Assert the GAE linearity identities per batch (slow)")
    args = parser.parse_args()

    probe = Gate2Probe(
        arm=args.arm,
        delta_raw=args.delta_raw,
        out_path=args.out,
        self_check=args.self_check,
    )
    probe.install()

    kwargs = load_regime_kwargs(args.config)
    kwargs.update(
        n_episodes=args.rounds,
        seed=args.seed,
        device=args.device,
        gpu_preload=args.gpu_preload,
        # use_telemetry MUST stay True: seed gradient stats are collected only
        # when telemetry is on (vectorized_trainer.py: collect_gradients =
        # use_telemetry and stride), and G2 hard-fails on unmeasured gradient
        # health (KTS-001, slot.py:_check_g2) => telemetry-off silently blocks
        # ALL blending/fossilization. Found the hard way (validate2-5).
        use_telemetry=True,
        telemetry_dir=str(args.out.with_suffix("")) + "_telemetry",
        quiet_analytics=True,
        # Match the n=5 instrumented RNG build (CRN pairing structure); the
        # injection consumes no RNG, so arms stay paired either way.
        rng_three_domain_split=True,
        # The n=5 runs RECORDED amp_enabled=false (TRAINING_STARTED, control_s41)
        # even though the config JSON says amp=true: on a bf16-capable local GPU
        # amp_dtype="auto" would silently enable bf16 host training, which changes
        # the seed-improvement gate signals (bf16 artifacts are quarantined per
        # docs/analysis/2026-06-15-bf16-artifact-quarantine.md). Pin to the
        # recorded regime: fp32.
        amp=False,
        amp_dtype="off",
        group_id=f"gate2_{args.arm}_s{args.seed}",
    )

    t0 = time.perf_counter()
    train_ppo_vectorized(**kwargs)
    wall_s = time.perf_counter() - t0

    summary = summarize(args.out)
    summary["arm"] = args.arm
    summary["seed"] = args.seed
    summary["delta_raw"] = args.delta_raw
    summary["wall_s"] = wall_s
    summary_path = args.out.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
