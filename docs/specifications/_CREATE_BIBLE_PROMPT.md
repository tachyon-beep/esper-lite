# Create Bible Prompt

Copy this prompt and give it to Claude Code directly (not via a subagent).

---

## Prompt Template

Replace `MODULE_NAME` with the target module (e.g., `tolaria`, `simic`, `kasmina`).

```
I need you to create a comprehensive Module Bible for the MODULE_NAME subsystem.

**CRITICAL: Read the skill first:**
Read `/home/john/.claude/skills/bible-maintenance/SKILL.md` - this defines MANDATORY protocols.

**STEP 1: File Assessment**
Run: `wc -l src/esper/MODULE_NAME/*.py | sort -n`
Create TodoWrite items for each file >500 lines.

**STEP 2: Dispatch Specialists (MANDATORY)**

You MUST dispatch these specialists IN PARALLEL using the Task tool:

For RL modules (simic):
- Task(subagent_type="drl-expert") - PPO, GAE, reward shaping analysis
- Task(subagent_type="pytorch-expert") - CUDA, tensor ops, memory

For PyTorch-heavy modules (kasmina, tolaria):
- Task(subagent_type="pytorch-expert") - gradient flow, hooks, CUDA

For contract/integration modules (leyline):
- Task(subagent_type="pytorch-expert") - tensor schemas
- Use axiom-python-engineering skill for protocol patterns

**STEP 3: Archive Existing Bible (if exists)**
If `docs/specifications/MODULE_NAME.md` exists:
1. Create archive: `docs/specifications/archive/MODULE_NAME-YYYY-MM-DD.md`
2. Move existing bible there
3. Then create fresh bible

**STEP 4: Wait for Specialists**
Do NOT proceed until all specialist Tasks return results.

**STEP 5: Create Bible**
- Template: `docs/specifications/_TEMPLATE.md`
- Output: `docs/specifications/MODULE_NAME.md`
- MUST include specialist insights in Tribal Knowledge with attribution
- MUST include file:line references

**STEP 6: Validation**
Run through the validation checklist in the skill.

**OUTPUT REQUIREMENTS:**
1. Show me the Task tool calls you made to specialists
2. Show me the specialist results you received
3. Show me how you incorporated their insights
```

---

## Quick Commands

### Tolaria (Training Loops)
```
Create a bible for Tolaria.
Read the skill at ~/.claude/skills/bible-maintenance/SKILL.md first.
Dispatch pytorch-expert for CUDA/optimizer/gradient analysis.
Archive existing bible if present, then create new one at docs/specifications/tolaria.md
```

### Simic (RL Infrastructure)
```
Create a bible for Simic.
Read the skill at ~/.claude/skills/bible-maintenance/SKILL.md first.
Dispatch BOTH drl-expert (PPO/GAE/rewards) AND pytorch-expert (CUDA/LSTM) in parallel.
Archive existing bible if present, then create new one at docs/specifications/simic.md
```

### Kasmina (Model/Slots)
```
Create a bible for Kasmina.
Read the skill at ~/.claude/skills/bible-maintenance/SKILL.md first.
Dispatch pytorch-expert for gradient isolation, hooks, and grafting analysis.
Archive existing bible if present, then create new one at docs/specifications/kasmina.md
```

### Leyline (Contracts)
```
Create a bible for Leyline.
Read the skill at ~/.claude/skills/bible-maintenance/SKILL.md first.
Dispatch pytorch-expert for tensor schema analysis.
Use axiom-python-engineering skill for Protocol/dataclass patterns.
Archive existing bible if present, then create new one at docs/specifications/leyline.md
```

### Nissa (Telemetry Hub)
```
Create a bible for Nissa.
Read the skill at ~/.claude/skills/bible-maintenance/SKILL.md first.
Check file sizes - dispatch specialists only if complex patterns found.
Archive existing bible if present, then create new one at docs/specifications/nissa.md
```

### Karn (Research Telemetry)
```
Create a bible for Karn.
Read the skill at ~/.claude/skills/bible-maintenance/SKILL.md first.
Check file sizes - dispatch specialists only if complex patterns found.
Archive existing bible if present, then create new one at docs/specifications/karn.md
```

---

## Archive Directory Setup

First-time setup (run once):
```bash
mkdir -p docs/specifications/archive
```
