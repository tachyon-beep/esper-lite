# Speculative Delta — Future Pruning & torch.ao Integration

Design anchor: docs/design/detailed_design/research_concepts/future_pytorch/pytorch-future-pruning.md

Summary
- Leverage PyTorch 2.8+ capabilities (semi‑structured 2:4 sparsity, torch.ao, FlashAttention‑3) to evolve checkpoint‑based pruning into a hardware‑aware optimisation stage, aligning Tezzeret/Urza pipelines with Elesh/Emrakul analyses.

Proposed Behaviour (Design‑style)
- Tezzeret: add a “Pruning/Optimisation” pipeline step at checkpoint time; record sparsity/quantisation metadata into Urza.
- Urza: store optimisation descriptors and checksums alongside artifacts; expose selection hints.
- Emrakul/Elesh: output structured masks and fold plans (heads/channels/layers) ready for torch.compile backends.

Status (Prototype)
- Implemented: None. Related design exists for checkpoint pruning (Kasmina 02.6), but no hardware‑aware integration.

Adoption (Low‑risk, prototype)
- Pilot 2:4 sparsity as metadata‑only: add a flag in artifact manifests and optional telemetry; leave model untouched. Prepare Tezzeret hooks without enabling transforms.

Cross‑Subsystem Impact
- Tezzeret/Urza (artifacts), Emrakul/Elesh (analysis), Kasmina (unchanged at runtime), Tamiyo/Nissa (telemetry visibility).

Implementation Tasks (Speculative)
- Urza manifest: Add optional optimisation descriptor fields (e.g., `sparsity_pattern`, `quantization_bits`, `backend_hint`).
- Tezzeret: Add a no‑op “optimisation” stage that records descriptors and compile duration; behind a feature flag.
- Nissa: Dashboards for compile durations and optimisation descriptors; track adoption over time.
- Emrakul/Elesh: Define structured masks/fold plans message shapes for future use (design only).
- Benchmarks: Add micro‑benchmarks plan for 2:4 sparsity (design doc + telemetry names), without enabling transforms yet.
