# Create Module Bible

Create a comprehensive Module Bible for the **$ARGUMENTS** subsystem following the bible-maintenance skill protocol.

## Required Steps

### 1. Read the Skill (MANDATORY)
Read and follow: `~/.claude/skills/bible-maintenance/SKILL.md`

### 2. Assess Files
```bash
wc -l src/esper/$ARGUMENTS/*.py | sort -n
```
Create TodoWrite items for ALL phases from the skill.

### 3. Archive Existing Bible (if exists)
If `docs/specifications/$ARGUMENTS.md` exists:
```bash
mkdir -p docs/specifications/archive
mv docs/specifications/$ARGUMENTS.md docs/specifications/archive/$ARGUMENTS-$(date +%Y-%m-%d).md
```

### 4. Dispatch Specialists (MANDATORY)
Based on module type, dispatch the appropriate specialists using the Task tool:

**For simic (RL):** Dispatch BOTH in parallel:
- `Task(subagent_type="drl-expert", prompt="Analyze src/esper/simic/ for PPO correctness, reward shaping validity, policy architecture, and DRL pitfalls...")`
- `Task(subagent_type="pytorch-expert", prompt="Analyze src/esper/simic/ for CUDA stream usage, LSTM patterns, memory management...")`

**For kasmina, tolaria (PyTorch-heavy):**
- `Task(subagent_type="pytorch-expert", prompt="Analyze src/esper/$ARGUMENTS/ for gradient flow, hooks, CUDA operations, memory patterns...")`

**For leyline (contracts):**
- `Task(subagent_type="pytorch-expert", prompt="Analyze tensor schemas in src/esper/leyline/...")`
- Use `Skill(axiom-python-engineering:using-python-engineering)` for Protocol patterns

**For nissa, karn (telemetry):** Check file sizes first, dispatch if >500 lines found.

### 5. Wait for Specialists
DO NOT proceed until all specialist Tasks complete and return results.

### 6. Create Bible
- Template: `docs/specifications/_TEMPLATE.md`
- Output: `docs/specifications/$ARGUMENTS.md`
- Include specialist insights with attribution in Tribal Knowledge
- Include file:line references for all gotchas

### 7. Validate
Run through the validation checklist from the skill.

## Success Criteria
- [ ] Skill was read and followed
- [ ] Specialists were ACTUALLY dispatched via Task tool
- [ ] Existing bible was archived (if present)
- [ ] New bible includes specialist insights with attribution
- [ ] Tribal Knowledge has ≥5 entries with file:line refs
