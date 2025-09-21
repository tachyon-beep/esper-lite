# Prototype Delta Rubric

Status categories:

| Status | Meaning |
| --- | --- |
| Implemented | Behaviour matches the design intent with test evidence. |
| Partially Implemented | Some behaviour present; gaps or simplifications remain. |
| Missing | Not present in the prototype implementation. |
| Diverges | Intentionally different from design (prototype simplification) or contradicts spec. |
| Not Applicable | Design item outside the prototype’s scope. |

Severity guidance:

| Severity | When to use | Typical action |
| --- | --- | --- |
| Must‑have | Required for safety/contract correctness or central claims | Prioritise for next slice |
| Should‑have | Important for robustness or parity | Plan near‑term |
| Nice‑to‑have | Useful but not critical | Triage later |

Evidence types:

| Type | Examples |
| --- | --- |
| Code reference | `src/esper/kasmina/lifecycle.py`, `src/esper/kasmina/seed_manager.py` |
| Test reference | `tests/kasmina/test_lifecycle.py`, `tests/integration/test_control_loop.py` |
| Absence | No module/function present; behaviour not exercised in tests |

Referencing style:
- Use repository‑relative file paths in backticks. Line numbers may be omitted for stability.
- Cite design sources by their path under `docs/design/detailed_design/`.

