"""Three-domain RNG split + compare-point for the causal-contribution harness.

Design §5.5 (split) + §5.6 (compare-point + determinism class). The estimand
the experiment exists to measure is corrupted by a subtle RNG bug: today a single
global generator is shared by three logically-distinct consumers
(controller sampling, host stochasticity, blueprint-init weight draws). Because
the rollout SAMPLES actions, suppressing a slot changes how many blueprint/host
draws happen between arms, OFFSETS the shared stream, and SHIFTS which actions
the controller samples thereafter — injecting a pure RNG artifact straight into
Δ_struct, the primary discriminator. This module gives each consumer its own
stream so the controller's sampling no longer depends on how many blueprint/host
draws occurred.

Gating: this object is ONLY constructed for the instrumented experiment build.
When it is absent, every call site threads ``generator=None`` (and skips the
blueprint-init fork), which is byte-identical to today's behavior. That is the
literal OFF gate — separate from the SUPPRESS-SLOT suppression toggle.

Determinism class: CRN / statistical (NOT bit-exact). See ``DeterminismClass``.

Three domains:
  (A) CONTROLLER — one ``torch.Generator`` on the policy device, seeded from the
      master seed. Threaded into ``get_action``'s op + per-head ``torch.multinomial``.
      Insulates controller sampling from host/blueprint draw counts. This is the
      load-bearing domain for the Δ_struct artifact.
  (B) HOST — a per-env ``torch.Generator`` for the germination shape-probe randn.
      HYGIENE ONLY: the probe output is a discarded shape smoke test
      (``slot.py:1393-1401``), so this domain's VALUES never touch training. It
      keeps the global default stream untouched and gives a clean separable
      domain for the compare-point. Confirmed for the R1 blueprint set
      (norm/lora/attention): the seed forwards contain NO dropout / MultiheadAttention
      dropout / randn / bernoulli (grep-verified over kasmina/blueprints), so the
      ONLY host-stochasticity consumer is the shape-probe — hence this generator
      advances only on a shape-probe cache MISS (one draw per (topology, channels)
      per slot), and host_state_hash must be read as a hygiene signal, not a
      training-trajectory fingerprint.
  (C) BLUEPRINT_INIT — content-addressed via a CPU default-generator
      save/seed/restore around ``BlueprintRegistry.create``. A germination's
      weights depend ONLY on its content seed, so a blueprint gets identical
      weights in both arms regardless of how many germinations preceded it (the
      only thing that survives the divergence point). Uses ``get_rng_state`` /
      ``default_generator.manual_seed`` / ``set_rng_state`` — CPU-only — NOT
      ``torch.manual_seed`` (which would also reseed every CUDA generator and leak
      straight into the experiment) and NOT bare ``fork_rng`` (which enumerates
      CUDA devices per call).
"""

from __future__ import annotations

import hashlib
from contextlib import contextmanager
from typing import Iterator

import torch

from esper.leyline import (
    CONTROLLER_RNG_SEED_OFFSET,
    HOST_RNG_SEED_OFFSET,
    INTERVENTION_DETERMINISM_CLASS,
)

# 63-bit mask keeps derived seeds in a safe, deterministic positive range.
_SEED_MASK = (1 << 63) - 1


def derive_child_seed(base_seed: int, offset: int) -> int:
    """Derive a deterministic child seed for one RNG domain.

    Folds the master/env seed with a per-domain offset and masks to 63 bits so
    the result is a stable, positive, generator-safe integer.
    """
    return (int(base_seed) + int(offset)) & _SEED_MASK


def _hash_state(state: torch.Tensor) -> str:
    """SHA-256 of a generator state ByteTensor (host-side, no CUDA sync)."""
    return hashlib.sha256(bytes(state.numpy().tobytes())).hexdigest()


class ExperimentRngDomains:
    """Owns the three split RNG streams + the compare-point.

    Constructed once per run from the master seed, AFTER controller init and
    data-order seeding (both of which key off the global default generator at
    ``vectorized.py:1148`` and the iterators' explicit ``seed``), so the pairing
    pillars (shared controller init, shared data order) are preserved: this
    object never calls ``torch.manual_seed`` mid-run.
    """

    def __init__(self, *, master_seed: int, controller_device: torch.device | str) -> None:
        self.master_seed = int(master_seed)
        self._controller_device = torch.device(controller_device)
        self.controller_generator = torch.Generator(device=self._controller_device)
        self.controller_generator.manual_seed(
            derive_child_seed(self.master_seed, CONTROLLER_RNG_SEED_OFFSET)
        )
        # Per-env host generators, created lazily so device placement matches the
        # env's device exactly. Insertion order == env order for a stable hash.
        self._host_generators: dict[int, torch.Generator] = {}
        # Cumulative count of content-addressed blueprint-init germinations.
        self._blueprint_germination_count = 0
        # Cumulative count of controller get_action invocations. Device-portable
        # (a plain Python int, no GPU sync); incremented once per rollout step via
        # record_controller_invocation(). This is the DISPOSITIVE anti-offset
        # signal: the controller draws a fixed number of times per step, so a
        # constant per-step DELTA within a run and equal cumulative counts across
        # paired arms prove the controller stream was never offset by blueprint/
        # host draws — without the cross-process generator-state-hash comparability
        # assumption a CUDA get_state() digest would carry.
        self._controller_draw_count = 0

    # -- domain (B): host stochasticity ------------------------------------
    def host_generator(
        self, env_idx: int, *, env_seed: int, device: torch.device | str
    ) -> torch.Generator:
        """Return (creating on first use) the per-env host generator.

        Deterministically seeded from the env seed so it is reproducible and
        independent of every other domain.
        """
        gen = self._host_generators.get(env_idx)
        if gen is None:
            gen = torch.Generator(device=torch.device(device))
            gen.manual_seed(derive_child_seed(env_seed, HOST_RNG_SEED_OFFSET))
            self._host_generators[env_idx] = gen
        return gen

    # -- domain (C): blueprint-init content addressing ---------------------
    @contextmanager
    def blueprint_init(self, content_seed: int) -> Iterator[None]:
        """Content-address CPU weight-init draws around a germination.

        Saves the CPU default-generator state, seeds it with the content seed,
        yields (the blueprint is constructed on CPU here), then restores the CPU
        state EXACTLY. The controller and host generators are untouched, and no
        CUDA generator is reseeded. The restore means blueprint-init draws do not
        advance any persistent stream — so they cannot offset the controller.
        """
        cpu_state = torch.get_rng_state()
        try:
            torch.default_generator.manual_seed(int(content_seed) & _SEED_MASK)
            yield
            # Count only a SUCCESSFUL germination (in try, after yield) so a
            # blueprint construction that raises does not inflate the count and
            # desync the compare-point across arms.
            self._blueprint_germination_count += 1
        finally:
            # Always restore the CPU default stream, even on a failed germination,
            # so a raising construction cannot leave the global generator reseeded.
            torch.set_rng_state(cpu_state)

    @property
    def blueprint_germination_count(self) -> int:
        return self._blueprint_germination_count

    # -- domain (A): controller draw accounting ----------------------------
    def record_controller_invocation(self) -> None:
        """Note one controller ``get_action`` call (one rollout step).

        Called by the trainer immediately after ``get_action``. The controller
        draws a fixed number of times per call, so this cumulative count is a
        device-portable proxy for the controller stream position.
        """
        self._controller_draw_count += 1

    @property
    def controller_draw_count(self) -> int:
        return self._controller_draw_count

    # -- compare-point (§5.6) ----------------------------------------------
    def controller_state_hash(self) -> str:
        """Digest of the controller generator state.

        WARNING — calls ``Generator.get_state()``, which is a GPU->CPU SYNC for a
        CUDA generator. Do NOT call every step in a long run (~120k syncs over
        200 episodes); the trainer emits this only at a fixed stride and at
        suppression events (the cheap per-step signal is ``controller_draw_count``).

        What it proves: pre-first-suppression parity across paired arms is
        guaranteed by the masks being IDENTICAL (the OFF hook mutates nothing and
        the controller stream is seeded identically) — equal state digests confirm
        that the controller drew the SAME NUMBER of times, i.e. no stream OFFSET.
        It is a draw-count / no-offset check, NOT a trajectory fingerprint: it
        cannot, on its own, localize "divergence onset coincides with the
        suppression event" (§5.6) because the digest is value-independent — that
        localization needs an action/trajectory hash, which is out of R1 scope.
        """
        return _hash_state(self.controller_generator.get_state())

    def host_state_hash(self) -> str:
        """Digest of the concatenated per-env host generator states (env order).

        Hygiene signal only (domain B): advances ONLY on a shape-probe cache miss
        (one draw per (topology, channels) per slot), and the probe output never
        touches training — so this is NOT a host-trajectory fingerprint.
        """
        if not self._host_generators:
            return _hash_state(torch.empty(0, dtype=torch.uint8))
        parts = [
            self._host_generators[idx].get_state()
            for idx in sorted(self._host_generators)
        ]
        return _hash_state(torch.cat(parts))

    def compare_point(self, *, include_state_hashes: bool = True) -> dict[str, object]:
        """Per-domain compare-point snapshot for the INTERVENTION_STEP payload.

        ``controller_draw_count`` + ``blueprint_draw_count`` are cheap host-side
        ints emitted EVERY step (the portable per-step parity signal). The two
        ``*_state_hash`` fields each cost a GPU->CPU sync, so they are populated
        only when ``include_state_hashes`` is True (the trainer strides them and
        forces them at suppression events); otherwise they are None.
        """
        snapshot: dict[str, object] = {
            "controller_draw_count": self._controller_draw_count,
            "blueprint_draw_count": self._blueprint_germination_count,
            "determinism_class": INTERVENTION_DETERMINISM_CLASS.value,
            "controller_state_hash": None,
            "host_state_hash": None,
        }
        if include_state_hashes:
            snapshot["controller_state_hash"] = self.controller_state_hash()
            snapshot["host_state_hash"] = self.host_state_hash()
        return snapshot


__all__ = [
    "ExperimentRngDomains",
    "derive_child_seed",
]
