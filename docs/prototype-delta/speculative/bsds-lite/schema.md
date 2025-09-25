# BSDS‑Lite JSON Schema (Prototype; Mirrors Leyline `BSDS`)

Fields (all optional unless noted). Consumers must ignore unknown fields. Canonical definitions live in Leyline as enums/messages (Day 1); these JSON fields mirror them for Urza extras transport. For automated validation, see the machine‑readable JSON Schema at `docs/prototype-delta/speculative/bsds-lite/schema.json`.

- `risk_score` (number, 0.0–1.0)
  - Overall risk in [0,1]; if present, takes precedence over descriptor `risk`.
- `hazard_band` (string)
  - One of: `LOW`, `MEDIUM`, `HIGH`, `CRITICAL` (mirrors Leyline `HazardBand`).
- `handling_class` (string)
  - One of: `standard`, `restricted`, `quarantine` (mirrors Leyline `HandlingClass`).
- `resource_profile` (string)
  - One of: `cpu`, `gpu`, `memory_heavy`, `io_heavy`, `mixed` (mirrors Leyline `ResourceProfile`).
- `mitigation` (object)
  - Optional keys: `recommendation` (string), `cooldown_ms` (number), `notes` (string).
- `provenance` (string)
  - Suggested values: `URABRASK`, `CURATED`, `HEURISTIC`, `EXTERNAL` (mirrors Leyline `Provenance`).
- `issued_at` (string)
  - RFC3339 timestamp.

JSON Schema (informal)
```
{
  "type": "object",
  "properties": {
    "risk_score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    "hazard_band": {"type": "string", "enum": ["LOW", "MEDIUM", "HIGH", "CRITICAL"]},
    "handling_class": {"type": "string", "enum": ["standard", "restricted", "quarantine"]},
    "resource_profile": {"type": "string", "enum": ["cpu", "gpu", "memory_heavy", "io_heavy", "mixed"]},
    "mitigation": {
      "type": "object",
      "properties": {
        "recommendation": {"type": "string"},
        "cooldown_ms": {"type": "number", "minimum": 0},
        "notes": {"type": "string"}
      },
      "additionalProperties": true
    },
    "provenance": {"type": "string"},
    "issued_at": {"type": "string"}
  },
  "additionalProperties": true
}
```

Producer Guidance
- Keep payloads < 4 KiB; avoid large arrays or nested blobs.
- Populate `provenance`; default to `HEURISTIC` until the crucible runs.
- Prefer stable field names and avoid renames; add new optional fields instead.

Consumer Guidance
- Treat absent fields as unknown/neutral; never crash on missing values.
- For gating, use `hazard_band` first, then `risk_score` as a numeric fallback.
