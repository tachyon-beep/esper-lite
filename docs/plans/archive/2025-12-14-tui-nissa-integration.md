# TUI/Nissa Integration + KARN P1 Completion

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Integrate Nissa event logging into TUI as a unified display, make TUI default, and complete remaining KARN P1 issues.

**Architecture:** TUI becomes the primary output interface with an integrated event log panel that captures ALL events (like ConsoleOutput does). ConsoleOutput becomes a fallback for non-TTY environments.

**Tech Stack:** Rich (Live, Layout, Panel), Python logging, threading for TUI state

---

## Tasks Overview

1. Add event log infrastructure to TUIState
2. Add event formatting method  
3. Add event log panel rendering
4. Update TUI layouts to include log panel
5. Buffer ALL events in emit()
6. Make TUI default, add --no-tui flag
7. Add TTY detection for graceful fallback
8. P1-01: Remove unused emit_to_nissa config
9. P1-04: Add CLI wiring for JSONL export
10. P1-05: Add Nissa DirectoryOutput importer
11. P1-06: Fix dense trace epoch vs episode handling
12. P1-07: Wire dashboard snapshot + add entropy field
13. P1-10: Normalize KL divergence naming
