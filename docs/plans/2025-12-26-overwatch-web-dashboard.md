# Overwatch Web Dashboard Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a Vue 3 + TypeScript web dashboard that mirrors Sanctum TUI functionality, receiving real-time `SanctumSnapshot` data over WebSocket.

**Architecture:** Server-side aggregation via existing `SanctumAggregator` (Python), broadcasting JSON snapshots at 10 Hz over WebSocket. Vue 3 frontend is a pure view layer with reactive state from `useOverwatch()` composable. Three-column layout: Anomaly sidebar | Main dashboard | Detail panel.

**Tech Stack:** Python (FastAPI + WebSocket), Vue 3 (Composition API), TypeScript, Vite, Apache ECharts (charts), CSS variables (observatory theme)

---

## Phase 1: Foundation

### Task 1: Create Vue 3 Project Scaffold

**Files:**
- Create: `src/esper/karn/overwatch/__init__.py`
- Create: `src/esper/karn/overwatch/web/package.json`
- Create: `src/esper/karn/overwatch/web/vite.config.ts`
- Create: `src/esper/karn/overwatch/web/tsconfig.json`
- Create: `src/esper/karn/overwatch/web/index.html`
- Create: `src/esper/karn/overwatch/web/src/main.ts`
- Create: `src/esper/karn/overwatch/web/src/App.vue`

**Step 1: Create Python package marker**

```python
# src/esper/karn/overwatch/__init__.py
"""Overwatch - Web-based training monitoring dashboard.

Overwatch provides a Vue 3 web interface for real-time training monitoring,
mirroring Sanctum TUI functionality with enhanced visualizations.

Usage:
    from esper.karn.overwatch import OverwatchBackend

    backend = OverwatchBackend(port=8080)
    hub.add_backend(backend)
"""

from esper.karn.overwatch.backend import OverwatchBackend

__all__ = ["OverwatchBackend"]
```

**Step 2: Create package.json**

```json
{
  "name": "overwatch",
  "version": "0.1.0",
  "private": true,
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "vue-tsc && vite build",
    "preview": "vite preview",
    "test": "vitest",
    "test:e2e": "playwright test",
    "generate:types": "cd ../../../.. && python scripts/generate_overwatch_types.py > src/esper/karn/overwatch/web/src/types/sanctum.ts"
  },
  "dependencies": {
    "vue": "^3.4.0",
    "echarts": "^5.5.0",
    "vue-echarts": "^6.6.0"
  },
  "devDependencies": {
    "@vitejs/plugin-vue": "^5.0.0",
    "@vue/test-utils": "^2.4.0",
    "typescript": "^5.3.0",
    "vite": "^5.0.0",
    "vitest": "^1.0.0",
    "vue-tsc": "^1.8.0",
    "playwright": "^1.40.0"
  }
}
```

**Step 3: Create vite.config.ts**

```typescript
// src/esper/karn/overwatch/web/vite.config.ts
import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

export default defineConfig({
  plugins: [vue()],
  server: {
    port: 3000,
    proxy: {
      '/ws': {
        target: 'ws://localhost:8080',
        ws: true
      }
    }
  },
  build: {
    outDir: 'dist',
    emptyOutDir: true
  }
})
```

**Step 4: Create tsconfig.json**

```json
{
  "compilerOptions": {
    "target": "ES2020",
    "useDefineForClassFields": true,
    "module": "ESNext",
    "lib": ["ES2020", "DOM", "DOM.Iterable"],
    "skipLibCheck": true,
    "moduleResolution": "bundler",
    "allowImportingTsExtensions": true,
    "resolveJsonModule": true,
    "isolatedModules": true,
    "noEmit": true,
    "jsx": "preserve",
    "strict": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "noFallthroughCasesInSwitch": true
  },
  "include": ["src/**/*.ts", "src/**/*.tsx", "src/**/*.vue"],
  "references": [{ "path": "./tsconfig.node.json" }]
}
```

**Step 5: Create index.html**

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Overwatch - Training Monitor</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Oxanium:wght@400;500;600;700&display=swap" rel="stylesheet">
</head>
<body>
  <div id="app"></div>
  <script type="module" src="/src/main.ts"></script>
</body>
</html>
```

**Step 6: Create main.ts**

```typescript
// src/esper/karn/overwatch/web/src/main.ts
import { createApp } from 'vue'
import App from './App.vue'
import './styles/theme.css'

createApp(App).mount('#app')
```

**Step 7: Create App.vue shell**

```vue
<!-- src/esper/karn/overwatch/web/src/App.vue -->
<script setup lang="ts">
import { ref } from 'vue'

const connectionState = ref<'connecting' | 'connected' | 'disconnected'>('connecting')
</script>

<template>
  <div class="overwatch">
    <header class="status-bar">
      <span class="status-indicator" :class="connectionState">
        {{ connectionState.toUpperCase() }}
      </span>
      <span class="title">OVERWATCH</span>
    </header>

    <main class="dashboard">
      <aside class="anomaly-sidebar">
        <!-- Anomaly sidebar -->
        <p>Anomalies</p>
      </aside>

      <section class="main-content">
        <!-- Main dashboard content -->
        <p>Dashboard loading...</p>
      </section>

      <aside class="detail-panel">
        <!-- Detail panel -->
        <p>Details</p>
      </aside>
    </main>
  </div>
</template>

<style scoped>
.overwatch {
  min-height: 100vh;
  display: flex;
  flex-direction: column;
  background: var(--bg-void);
  color: var(--text-primary);
}

.status-bar {
  display: flex;
  align-items: center;
  gap: var(--space-md);
  padding: var(--space-sm) var(--space-md);
  background: var(--bg-panel);
  border-bottom: 1px solid var(--border-subtle);
}

.status-indicator {
  font-size: 10px;
  font-weight: 600;
  padding: 2px 8px;
  border-radius: 2px;
}

.status-indicator.connecting {
  background: var(--status-warn);
  color: var(--bg-void);
}

.status-indicator.connected {
  background: var(--status-win);
  color: var(--bg-void);
}

.status-indicator.disconnected {
  background: var(--status-loss);
  color: var(--bg-void);
}

.title {
  font-family: var(--font-display);
  font-size: 13px;
  font-weight: 600;
  letter-spacing: 3px;
  color: var(--glow-cyan);
}

.dashboard {
  flex: 1;
  display: grid;
  grid-template-columns: 200px 1fr 300px;
  gap: var(--space-md);
  padding: var(--space-md);
}

.anomaly-sidebar,
.detail-panel {
  background: var(--bg-panel);
  border: 1px solid var(--border-subtle);
  border-radius: 4px;
  padding: var(--space-md);
}

.main-content {
  display: flex;
  flex-direction: column;
  gap: var(--space-md);
}
</style>
```

**Step 8: Verify scaffold builds**

Run:
```bash
cd src/esper/karn/overwatch/web && npm install && npm run build
```
Expected: Build completes with dist/ folder created

**Step 9: Commit**

```bash
git add src/esper/karn/overwatch/
git commit -m "feat(overwatch): create Vue 3 project scaffold"
```

---

### Task 2: Create Observatory Theme CSS

**Files:**
- Create: `src/esper/karn/overwatch/web/src/styles/theme.css`

**Step 1: Extract and create theme**

```css
/* src/esper/karn/overwatch/web/src/styles/theme.css */
/* Observatory Theme - extracted from dashboard.html */

:root {
  /* Deep observatory background */
  --bg-void: #05080f;
  --bg-primary: #0a1020;
  --bg-panel: #0d1528;
  --bg-elevated: #121d35;

  /* Bioluminescent accents */
  --glow-cyan: #00e5ff;
  --glow-cyan-dim: rgba(0, 229, 255, 0.15);
  --glow-cyan-bright: rgba(0, 229, 255, 0.4);

  /* Status colors */
  --status-win: #00ff9d;
  --status-win-glow: rgba(0, 255, 157, 0.3);
  --status-loss: #ff5c5c;
  --status-loss-glow: rgba(255, 92, 92, 0.3);
  --status-warn: #ffb347;
  --status-warn-glow: rgba(255, 179, 71, 0.3);
  --status-neutral: #6b7c9e;

  /* Seed stage colors */
  --stage-dormant: #3d4a66;
  --stage-germinated: #7c4dff;
  --stage-training: #00b8d4;
  --stage-blending: #ffab00;
  --stage-holding: #ff6d00;
  --stage-fossilized: #00e676;
  --stage-pruned: #ff5252;

  /* Text hierarchy */
  --text-bright: #f0f6ff;
  --text-primary: #c8d4e8;
  --text-secondary: #6b7c9e;
  --text-dim: #3d4a66;

  /* Borders */
  --border-subtle: rgba(100, 130, 180, 0.15);
  --border-glow: rgba(0, 229, 255, 0.3);

  /* Typography */
  --font-display: 'Oxanium', sans-serif;
  --font-mono: 'JetBrains Mono', monospace;

  /* Spacing */
  --space-xs: 4px;
  --space-sm: 8px;
  --space-md: 16px;
  --space-lg: 24px;
  --space-xl: 32px;
}

* {
  box-sizing: border-box;
  margin: 0;
  padding: 0;
}

body {
  font-family: var(--font-mono);
  background: var(--bg-void);
  color: var(--text-primary);
  line-height: 1.5;

  /* Subtle grid texture */
  background-image:
    linear-gradient(var(--bg-primary) 1px, transparent 1px),
    linear-gradient(90deg, var(--bg-primary) 1px, transparent 1px);
  background-size: 40px 40px;
}

/* Focus styles for accessibility */
:focus-visible {
  outline: 2px solid var(--glow-cyan);
  outline-offset: 2px;
}

/* Scrollbar styling */
::-webkit-scrollbar {
  width: 8px;
  height: 8px;
}

::-webkit-scrollbar-track {
  background: var(--bg-primary);
}

::-webkit-scrollbar-thumb {
  background: var(--border-subtle);
  border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
  background: var(--text-dim);
}
```

**Step 2: Verify theme applies**

Run:
```bash
cd src/esper/karn/overwatch/web && npm run dev
```
Expected: Dev server starts, page shows dark observatory theme

**Step 3: Commit**

```bash
git add src/esper/karn/overwatch/web/src/styles/
git commit -m "feat(overwatch): add observatory theme CSS"
```

---

### Task 3: Generate TypeScript Types from Python Schema

**Files:**
- Create: `scripts/generate_overwatch_types.py`
- Create: `src/esper/karn/overwatch/web/src/types/sanctum.ts`

**Step 1: Create type generator script**

```python
#!/usr/bin/env python3
"""Generate TypeScript interfaces from Sanctum schema dataclasses.

Usage:
    python scripts/generate_overwatch_types.py > src/esper/karn/overwatch/web/src/types/sanctum.ts
"""

from __future__ import annotations

import dataclasses
import sys
from datetime import datetime
from typing import get_type_hints, get_origin, get_args, Union

# Add src to path for imports
sys.path.insert(0, "src")

from esper.karn.sanctum.schema import (
    SanctumSnapshot,
    EnvState,
    SeedState,
    TamiyoState,
    CounterfactualSnapshot,
    CounterfactualConfig,
    DecisionSnapshot,
    AnomalyEvent,
)
from esper.leyline import SeedStage


def python_to_ts_type(py_type: type) -> str:
    """Convert Python type annotation to TypeScript type."""
    origin = get_origin(py_type)
    args = get_args(py_type)

    # Handle None/Optional
    if py_type is type(None):
        return "null"

    # Handle Union (includes Optional)
    if origin is Union:
        ts_types = [python_to_ts_type(arg) for arg in args]
        return " | ".join(ts_types)

    # Handle list/tuple
    if origin in (list, tuple):
        if args:
            inner = python_to_ts_type(args[0])
            return f"{inner}[]"
        return "unknown[]"

    # Handle dict
    if origin is dict:
        if len(args) == 2:
            key_type = python_to_ts_type(args[0])
            val_type = python_to_ts_type(args[1])
            return f"Record<{key_type}, {val_type}>"
        return "Record<string, unknown>"

    # Handle deque as array
    if hasattr(py_type, "__origin__") and "deque" in str(py_type):
        return "unknown[]"

    # Primitive mappings
    type_map = {
        str: "string",
        int: "number",
        float: "number",
        bool: "boolean",
        datetime: "string",  # ISO string
        type(None): "null",
    }

    if py_type in type_map:
        return type_map[py_type]

    # Check if it's an enum
    if hasattr(py_type, "__members__"):
        return py_type.__name__

    # Check if it's a dataclass we know
    if dataclasses.is_dataclass(py_type):
        return py_type.__name__

    # Fallback
    return "unknown"


def generate_enum(enum_cls: type) -> str:
    """Generate TypeScript type for Python enum."""
    members = [f'"{m.name}"' for m in enum_cls]
    return f"export type {enum_cls.__name__} = {' | '.join(members)};"


def generate_interface(cls: type) -> str:
    """Generate TypeScript interface from Python dataclass."""
    if not dataclasses.is_dataclass(cls):
        raise ValueError(f"{cls} is not a dataclass")

    lines = [f"export interface {cls.__name__} {{"]

    hints = get_type_hints(cls)
    for field in dataclasses.fields(cls):
        ts_type = python_to_ts_type(hints.get(field.name, field.type))
        lines.append(f"  {field.name}: {ts_type};")

    lines.append("}")
    return "\n".join(lines)


def main() -> None:
    """Generate all TypeScript types."""
    print("// Auto-generated from Python schema - DO NOT EDIT")
    print("// Run: python scripts/generate_overwatch_types.py")
    print()

    # Generate enums
    print(generate_enum(SeedStage))
    print()

    # Generate interfaces in dependency order
    dataclasses_to_generate = [
        CounterfactualConfig,
        CounterfactualSnapshot,
        SeedState,
        DecisionSnapshot,
        AnomalyEvent,
        TamiyoState,
        EnvState,
        SanctumSnapshot,
    ]

    for cls in dataclasses_to_generate:
        print(generate_interface(cls))
        print()


if __name__ == "__main__":
    main()
```

**Step 2: Run generator and create types file**

Run:
```bash
PYTHONPATH=src python scripts/generate_overwatch_types.py > src/esper/karn/overwatch/web/src/types/sanctum.ts
```
Expected: TypeScript file created with all interfaces

**Step 3: Verify types compile**

Run:
```bash
cd src/esper/karn/overwatch/web && npx vue-tsc --noEmit
```
Expected: No TypeScript errors

**Step 4: Commit**

```bash
git add scripts/generate_overwatch_types.py src/esper/karn/overwatch/web/src/types/
git commit -m "feat(overwatch): add TypeScript type generator"
```

---

### Task 4: Create useOverwatch Composable

**Files:**
- Create: `src/esper/karn/overwatch/web/src/composables/useOverwatch.ts`
- Create: `src/esper/karn/overwatch/web/src/composables/__tests__/useOverwatch.spec.ts`

**Step 1: Write the failing test**

```typescript
// src/esper/karn/overwatch/web/src/composables/__tests__/useOverwatch.spec.ts
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { useOverwatch } from '../useOverwatch'
import { nextTick } from 'vue'

// Mock WebSocket
class MockWebSocket {
  static instances: MockWebSocket[] = []
  onopen: (() => void) | null = null
  onmessage: ((event: { data: string }) => void) | null = null
  onclose: (() => void) | null = null
  onerror: ((error: Event) => void) | null = null
  readyState = 0

  constructor(public url: string) {
    MockWebSocket.instances.push(this)
  }

  close() {
    this.readyState = 3
  }

  simulateOpen() {
    this.readyState = 1
    this.onopen?.()
  }

  simulateMessage(data: object) {
    this.onmessage?.({ data: JSON.stringify(data) })
  }

  simulateClose() {
    this.readyState = 3
    this.onclose?.()
  }
}

describe('useOverwatch', () => {
  beforeEach(() => {
    MockWebSocket.instances = []
    vi.stubGlobal('WebSocket', MockWebSocket)
  })

  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('starts in connecting state', () => {
    const { connectionState } = useOverwatch('ws://localhost:8080/ws')
    expect(connectionState.value).toBe('connecting')
  })

  it('transitions to connected when WebSocket opens', async () => {
    const { connectionState } = useOverwatch('ws://localhost:8080/ws')

    MockWebSocket.instances[0].simulateOpen()
    await nextTick()

    expect(connectionState.value).toBe('connected')
  })

  it('updates snapshot when message received', async () => {
    const { snapshot } = useOverwatch('ws://localhost:8080/ws')

    const ws = MockWebSocket.instances[0]
    ws.simulateOpen()
    ws.simulateMessage({
      type: 'snapshot',
      data: { episode: 42, epoch: 10 }
    })
    await nextTick()

    expect(snapshot.value?.episode).toBe(42)
    expect(snapshot.value?.epoch).toBe(10)
  })

  it('tracks staleness', async () => {
    vi.useFakeTimers()
    const { staleness, lastUpdate } = useOverwatch('ws://localhost:8080/ws')

    const ws = MockWebSocket.instances[0]
    ws.simulateOpen()
    ws.simulateMessage({ type: 'snapshot', data: { episode: 1 } })
    await nextTick()

    expect(lastUpdate.value).toBeGreaterThan(0)

    vi.advanceTimersByTime(5000)
    expect(staleness.value).toBeGreaterThanOrEqual(5000)

    vi.useRealTimers()
  })
})
```

**Step 2: Run test to verify it fails**

Run:
```bash
cd src/esper/karn/overwatch/web && npm run test -- useOverwatch.spec.ts
```
Expected: FAIL with "Cannot find module '../useOverwatch'"

**Step 3: Write the composable**

```typescript
// src/esper/karn/overwatch/web/src/composables/useOverwatch.ts
import { ref, computed, onUnmounted, type Ref, type ComputedRef } from 'vue'
import type { SanctumSnapshot } from '../types/sanctum'

export type ConnectionState = 'connecting' | 'connected' | 'disconnected'

export interface UseOverwatchReturn {
  snapshot: Ref<SanctumSnapshot | null>
  connectionState: Ref<ConnectionState>
  lastUpdate: Ref<number>
  staleness: ComputedRef<number>
  reconnect: () => void
}

export function useOverwatch(url: string): UseOverwatchReturn {
  const snapshot = ref<SanctumSnapshot | null>(null)
  const connectionState = ref<ConnectionState>('connecting')
  const lastUpdate = ref<number>(0)

  let ws: WebSocket | null = null
  let reconnectTimeout: ReturnType<typeof setTimeout> | null = null
  let stalenessInterval: ReturnType<typeof setInterval> | null = null

  // Track staleness reactively
  const now = ref(Date.now())
  stalenessInterval = setInterval(() => {
    now.value = Date.now()
  }, 100)

  const staleness = computed(() => {
    if (lastUpdate.value === 0) return 0
    return now.value - lastUpdate.value
  })

  function connect() {
    connectionState.value = 'connecting'
    ws = new WebSocket(url)

    ws.onopen = () => {
      connectionState.value = 'connected'
    }

    ws.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data)
        if (message.type === 'snapshot') {
          snapshot.value = message.data
          lastUpdate.value = Date.now()
        }
      } catch (e) {
        console.error('Failed to parse WebSocket message:', e)
      }
    }

    ws.onclose = () => {
      connectionState.value = 'disconnected'
      // Auto-reconnect after 2 seconds
      reconnectTimeout = setTimeout(connect, 2000)
    }

    ws.onerror = () => {
      ws?.close()
    }
  }

  function reconnect() {
    if (reconnectTimeout) {
      clearTimeout(reconnectTimeout)
      reconnectTimeout = null
    }
    ws?.close()
    connect()
  }

  // Initial connection
  connect()

  // Cleanup on unmount
  onUnmounted(() => {
    if (reconnectTimeout) clearTimeout(reconnectTimeout)
    if (stalenessInterval) clearInterval(stalenessInterval)
    ws?.close()
  })

  return {
    snapshot,
    connectionState,
    lastUpdate,
    staleness,
    reconnect
  }
}
```

**Step 4: Run test to verify it passes**

Run:
```bash
cd src/esper/karn/overwatch/web && npm run test -- useOverwatch.spec.ts
```
Expected: All tests PASS

**Step 5: Commit**

```bash
git add src/esper/karn/overwatch/web/src/composables/
git commit -m "feat(overwatch): add useOverwatch WebSocket composable"
```

---

### Task 5: Create OverwatchBackend Python Server

**Files:**
- Create: `src/esper/karn/overwatch/backend.py`
- Create: `tests/karn/overwatch/test_backend.py`

**Step 1: Write the failing test**

```python
# tests/karn/overwatch/test_backend.py
"""Tests for OverwatchBackend WebSocket server."""

import json
import pytest
from unittest.mock import MagicMock, patch

from esper.karn.overwatch.backend import OverwatchBackend
from esper.karn.sanctum.schema import SanctumSnapshot


class TestOverwatchBackend:
    """Test OverwatchBackend initialization and event processing."""

    def test_backend_initializes_with_aggregator(self):
        """Backend should create a SanctumAggregator instance."""
        backend = OverwatchBackend(port=8080)
        assert backend.aggregator is not None
        assert backend.port == 8080

    def test_emit_processes_event_through_aggregator(self):
        """Events should be passed to the aggregator."""
        backend = OverwatchBackend(port=8080)
        backend.aggregator = MagicMock()

        mock_event = MagicMock()
        backend.emit(mock_event)

        backend.aggregator.process_event.assert_called_once_with(mock_event)

    def test_snapshot_to_json_produces_valid_json(self):
        """Snapshots should serialize to valid JSON."""
        backend = OverwatchBackend(port=8080)

        # Get a snapshot (will be empty/default)
        snapshot = backend.get_snapshot()
        json_str = backend.snapshot_to_json(snapshot)

        # Should be valid JSON
        parsed = json.loads(json_str)
        assert "episode" in parsed
        assert "epoch" in parsed

    def test_rate_limiting_throttles_broadcasts(self):
        """Should not broadcast more than configured Hz."""
        backend = OverwatchBackend(port=8080, snapshot_rate_hz=10)
        backend._broadcast = MagicMock()

        # Emit 50 events rapidly
        for _ in range(50):
            backend.emit(MagicMock())
            backend.maybe_broadcast()

        # At 10 Hz over ~0 seconds, should have at most 1-2 broadcasts
        assert backend._broadcast.call_count <= 5
```

**Step 2: Run test to verify it fails**

Run:
```bash
PYTHONPATH=src uv run pytest tests/karn/overwatch/test_backend.py -v
```
Expected: FAIL with "No module named 'esper.karn.overwatch.backend'"

**Step 3: Write the backend**

```python
# src/esper/karn/overwatch/backend.py
"""OverwatchBackend - WebSocket server for web dashboard.

Wraps SanctumAggregator and broadcasts snapshots via WebSocket.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from dataclasses import asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from queue import Queue, Empty
from typing import TYPE_CHECKING, Any

from esper.karn.sanctum.aggregator import SanctumAggregator
from esper.karn.sanctum.schema import SanctumSnapshot

if TYPE_CHECKING:
    from esper.karn.contracts import TelemetryEventLike

_logger = logging.getLogger(__name__)


def _json_serializer(obj: Any) -> Any:
    """Custom JSON serializer for dataclasses and special types."""
    if isinstance(obj, Enum):
        return obj.name
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


class OverwatchBackend:
    """WebSocket server broadcasting SanctumSnapshots.

    Implements OutputBackend protocol for integration with Nissa hub.

    Args:
        port: WebSocket server port (default: 8080)
        host: Server host (default: 0.0.0.0)
        snapshot_rate_hz: Maximum snapshot broadcast rate (default: 10)

    Usage:
        backend = OverwatchBackend(port=8080)
        backend.start()

        # In training loop:
        hub.add_backend(backend)
    """

    def __init__(
        self,
        port: int = 8080,
        host: str = "0.0.0.0",
        snapshot_rate_hz: float = 10.0,
    ):
        self.port = port
        self.host = host
        self.snapshot_rate_hz = snapshot_rate_hz
        self._min_interval = 1.0 / snapshot_rate_hz

        self.aggregator = SanctumAggregator()

        self._last_broadcast_time = 0.0
        self._broadcast_queue: Queue[str] = Queue(maxsize=100)
        self._clients: set[Any] = set()
        self._thread: threading.Thread | None = None
        self._running = False
        self._ready = threading.Event()

    def emit(self, event: "TelemetryEventLike") -> None:
        """Process telemetry event (OutputBackend interface)."""
        self.aggregator.process_event(event)

    def maybe_broadcast(self) -> None:
        """Broadcast snapshot if rate limit allows."""
        now = time.monotonic()
        if now - self._last_broadcast_time >= self._min_interval:
            self._broadcast_snapshot()
            self._last_broadcast_time = now

    def _broadcast_snapshot(self) -> None:
        """Queue snapshot for broadcast to all clients."""
        snapshot = self.get_snapshot()
        json_str = self.snapshot_to_json(snapshot)

        message = json.dumps({
            "type": "snapshot",
            "data": json.loads(json_str),
            "timestamp": time.time(),
        })

        try:
            self._broadcast_queue.put_nowait(message)
        except Exception:
            pass  # Drop if queue full

    def _broadcast(self, message: str) -> None:
        """Broadcast message to all connected clients."""
        # Implemented by async server loop
        pass

    def get_snapshot(self) -> SanctumSnapshot:
        """Get current aggregated snapshot."""
        return self.aggregator.get_snapshot()

    def snapshot_to_json(self, snapshot: SanctumSnapshot) -> str:
        """Serialize snapshot to JSON string."""
        return json.dumps(asdict(snapshot), default=_json_serializer)

    def start(self) -> None:
        """Start WebSocket server in background thread."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._run_server, daemon=True)
        self._thread.start()
        self._ready.wait(timeout=5.0)

    def stop(self) -> None:
        """Stop WebSocket server."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def _run_server(self) -> None:
        """Run async server in thread."""
        asyncio.run(self._async_server())

    async def _async_server(self) -> None:
        """Async WebSocket server implementation."""
        try:
            from fastapi import FastAPI, WebSocket, WebSocketDisconnect
            from fastapi.staticfiles import StaticFiles
            from fastapi.responses import FileResponse
            import uvicorn
        except ImportError:
            _logger.error("FastAPI not installed. Run: pip install esper-lite[dashboard]")
            return

        app = FastAPI(title="Overwatch Dashboard")

        # Serve static files from web/dist
        static_dir = Path(__file__).parent / "web" / "dist"
        if static_dir.exists():
            app.mount("/assets", StaticFiles(directory=static_dir / "assets"), name="assets")

            @app.get("/")
            async def serve_index():
                return FileResponse(static_dir / "index.html")

        @app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            self._clients.add(websocket)

            try:
                # Send initial snapshot
                snapshot = self.get_snapshot()
                await websocket.send_json({
                    "type": "connected",
                    "serverVersion": "1.0.0",
                    "snapshotRate": self.snapshot_rate_hz,
                })

                # Broadcast loop
                while self._running:
                    try:
                        message = self._broadcast_queue.get(timeout=0.1)
                        await websocket.send_text(message)
                    except Empty:
                        continue

            except WebSocketDisconnect:
                pass
            finally:
                self._clients.discard(websocket)

        self._ready.set()

        config = uvicorn.Config(
            app,
            host=self.host,
            port=self.port,
            log_level="warning",
        )
        server = uvicorn.Server(config)
        await server.serve()
```

**Step 4: Create __init__.py for tests**

```python
# tests/karn/overwatch/__init__.py
"""Tests for Overwatch web dashboard."""
```

**Step 5: Run test to verify it passes**

Run:
```bash
PYTHONPATH=src uv run pytest tests/karn/overwatch/test_backend.py -v
```
Expected: All tests PASS

**Step 6: Commit**

```bash
git add src/esper/karn/overwatch/backend.py tests/karn/overwatch/
git commit -m "feat(overwatch): add OverwatchBackend WebSocket server"
```

---

### Task 6: Add --overwatch CLI Flag

**Files:**
- Modify: `src/esper/scripts/train.py`
- Modify: `tests/scripts/test_train_sanctum_flag.py`

**Step 1: Write the failing test**

```python
# Add to tests/scripts/test_train_sanctum_flag.py

def test_overwatch_flag_exists(self):
    """--overwatch flag should be recognized."""
    result = subprocess.run(
        [sys.executable, "-m", "esper.scripts.train", "ppo", "--help"],
        capture_output=True,
        text=True,
    )
    assert "--overwatch" in result.stdout

def test_overwatch_and_sanctum_mutual_exclusion(self):
    """--sanctum and --overwatch together should error."""
    result = subprocess.run(
        [
            sys.executable, "-m", "esper.scripts.train",
            "ppo", "--overwatch", "--sanctum",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "mutually exclusive" in result.stderr.lower()
```

**Step 2: Run test to verify it fails**

Run:
```bash
PYTHONPATH=src uv run pytest tests/scripts/test_train_sanctum_flag.py::TestSanctumCLIFlag::test_overwatch_flag_exists -v
```
Expected: FAIL with assertion error

**Step 3: Add --overwatch flag to train.py**

Find the argument parser section and add:

```python
# After --sanctum definition
parser.add_argument(
    "--overwatch",
    action="store_true",
    help="Launch Overwatch web dashboard (mutually exclusive with --sanctum)",
)
parser.add_argument(
    "--overwatch-port",
    type=int,
    default=8080,
    help="Overwatch dashboard port (default: 8080)",
)
```

Add mutual exclusion check after argument parsing:

```python
if args.sanctum and args.overwatch:
    parser.error("--sanctum and --overwatch are mutually exclusive")
```

Add backend setup in the training initialization:

```python
if args.overwatch:
    from esper.karn.overwatch import OverwatchBackend
    overwatch_backend = OverwatchBackend(port=args.overwatch_port)
    overwatch_backend.start()
    hub.add_backend(overwatch_backend)
    console.print(f"[cyan]Overwatch dashboard: http://localhost:{args.overwatch_port}[/cyan]")
```

**Step 4: Run test to verify it passes**

Run:
```bash
PYTHONPATH=src uv run pytest tests/scripts/test_train_sanctum_flag.py -v
```
Expected: All tests PASS

**Step 5: Commit**

```bash
git add src/esper/scripts/train.py tests/scripts/test_train_sanctum_flag.py
git commit -m "feat(overwatch): add --overwatch CLI flag"
```

---

## Phase 2: Core Monitoring Widgets

### Task 7: Create StatusBar Component

**Files:**
- Create: `src/esper/karn/overwatch/web/src/components/StatusBar.vue`
- Create: `src/esper/karn/overwatch/web/src/components/__tests__/StatusBar.spec.ts`

**Step 1: Write the failing test**

```typescript
// src/esper/karn/overwatch/web/src/components/__tests__/StatusBar.spec.ts
import { describe, it, expect } from 'vitest'
import { mount } from '@vue/test-utils'
import StatusBar from '../StatusBar.vue'

describe('StatusBar', () => {
  it('displays connection state', () => {
    const wrapper = mount(StatusBar, {
      props: {
        connectionState: 'connected',
        staleness: 100,
        episode: 42,
        epoch: 10,
        batch: 5
      }
    })

    expect(wrapper.find('[data-testid="connection-status"]').text()).toContain('CONNECTED')
  })

  it('shows staleness warning when stale', () => {
    const wrapper = mount(StatusBar, {
      props: {
        connectionState: 'connected',
        staleness: 5000,  // 5 seconds
        episode: 42,
        epoch: 10,
        batch: 5
      }
    })

    expect(wrapper.find('[data-testid="staleness"]').classes()).toContain('warning')
  })

  it('displays episode and epoch', () => {
    const wrapper = mount(StatusBar, {
      props: {
        connectionState: 'connected',
        staleness: 100,
        episode: 42,
        epoch: 10,
        batch: 5
      }
    })

    expect(wrapper.text()).toContain('Ep 42')
    expect(wrapper.text()).toContain('Epoch 10')
  })
})
```

**Step 2: Run test to verify it fails**

Run:
```bash
cd src/esper/karn/overwatch/web && npm run test -- StatusBar.spec.ts
```
Expected: FAIL

**Step 3: Implement StatusBar component**

```vue
<!-- src/esper/karn/overwatch/web/src/components/StatusBar.vue -->
<script setup lang="ts">
import { computed } from 'vue'

const props = defineProps<{
  connectionState: 'connecting' | 'connected' | 'disconnected'
  staleness: number
  episode: number
  epoch: number
  batch: number
}>()

const stalenessText = computed(() => {
  if (props.staleness < 1000) return 'Just now'
  if (props.staleness < 10000) return `${(props.staleness / 1000).toFixed(1)}s ago`
  return 'STALE'
})

const stalenessClass = computed(() => {
  if (props.staleness < 2000) return 'fresh'
  if (props.staleness < 10000) return 'warning'
  return 'stale'
})
</script>

<template>
  <header class="status-bar">
    <div class="status-section">
      <span
        class="status-indicator"
        :class="connectionState"
        data-testid="connection-status"
      >
        {{ connectionState.toUpperCase() }}
      </span>
      <span
        class="staleness"
        :class="stalenessClass"
        data-testid="staleness"
      >
        {{ stalenessText }}
      </span>
    </div>

    <span class="title">OVERWATCH</span>

    <div class="metrics-section">
      <span class="metric">Ep {{ episode }}</span>
      <span class="separator">|</span>
      <span class="metric">Epoch {{ epoch }}</span>
      <span class="separator">|</span>
      <span class="metric">Batch {{ batch }}</span>
    </div>
  </header>
</template>

<style scoped>
.status-bar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: var(--space-sm) var(--space-md);
  background: var(--bg-panel);
  border-bottom: 1px solid var(--border-subtle);
  font-size: 12px;
}

.status-section {
  display: flex;
  align-items: center;
  gap: var(--space-md);
}

.status-indicator {
  font-size: 10px;
  font-weight: 600;
  padding: 2px 8px;
  border-radius: 2px;
}

.status-indicator.connecting {
  background: var(--status-warn);
  color: var(--bg-void);
}

.status-indicator.connected {
  background: var(--status-win);
  color: var(--bg-void);
}

.status-indicator.disconnected {
  background: var(--status-loss);
  color: var(--bg-void);
}

.staleness {
  color: var(--text-secondary);
}

.staleness.warning {
  color: var(--status-warn);
}

.staleness.stale {
  color: var(--status-loss);
  animation: pulse 1s infinite;
}

@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.5; }
}

.title {
  font-family: var(--font-display);
  font-size: 13px;
  font-weight: 600;
  letter-spacing: 3px;
  color: var(--glow-cyan);
}

.metrics-section {
  display: flex;
  align-items: center;
  gap: var(--space-sm);
}

.metric {
  color: var(--text-primary);
}

.separator {
  color: var(--text-dim);
}
</style>
```

**Step 4: Run test to verify it passes**

Run:
```bash
cd src/esper/karn/overwatch/web && npm run test -- StatusBar.spec.ts
```
Expected: All tests PASS

**Step 5: Commit**

```bash
git add src/esper/karn/overwatch/web/src/components/
git commit -m "feat(overwatch): add StatusBar component"
```

---

### Task 8-17: Additional Components

*[Remaining tasks follow the same TDD pattern for:]*

- **Task 8:** EnvironmentGrid.vue - Visual environment matrix
- **Task 9:** SeedChip.vue - Reusable seed status chip
- **Task 10:** AnomalySidebar.vue - Persistent anomaly list
- **Task 11:** LeaderboardTable.vue - Sortable scoreboard
- **Task 12:** HealthGauges.vue - Circular gauges with sparklines
- **Task 13:** SeedSwimlane.vue - Lifecycle timeline (ECharts)
- **Task 14:** ContributionWaterfall.vue - SVG waterfall chart
- **Task 15:** PolicyDiagnostics.vue - Per-head entropy breakdown
- **Task 16:** GradientHeatmap.vue - Layer × time heatmap
- **Task 17:** EventTimeline.vue - Virtual-scroll event log

*Each task follows the pattern:*
1. Write failing test
2. Run test, verify failure
3. Implement component
4. Run test, verify pass
5. Commit

---

## Phase 3: Integration & Polish

### Task 18: Wire Components in App.vue

**Files:**
- Modify: `src/esper/karn/overwatch/web/src/App.vue`

*[Connect all components to useOverwatch composable]*

---

### Task 19: Add Keyboard Navigation

**Files:**
- Create: `src/esper/karn/overwatch/web/src/composables/useKeyboard.ts`

*[Port TUI keyboard shortcuts: j/k, 1-9, /, Esc, etc.]*

---

### Task 20: E2E Tests with Playwright

**Files:**
- Create: `src/esper/karn/overwatch/web/e2e/dashboard.spec.ts`

*[Full integration tests with mock training server]*

---

### Task 21: Update pyproject.toml

**Files:**
- Modify: `pyproject.toml`

Add overwatch optional dependency back:

```toml
[project.optional-dependencies]
overwatch = [
    "fastapi>=0.115.0",
    "uvicorn[standard]>=0.32.0",
]
```

---

### Task 22: Update README Documentation

**Files:**
- Modify: `README.md`

*[Add --overwatch flag to CLI reference, document web dashboard usage]*

---

## Verification

After all tasks complete:

```bash
# Python tests
PYTHONPATH=src uv run pytest tests/karn/overwatch/ -v

# Frontend tests
cd src/esper/karn/overwatch/web && npm run test

# Build
cd src/esper/karn/overwatch/web && npm run build

# Integration test
PYTHONPATH=src python -m esper.scripts.train ppo --overwatch --episodes 1
# Open http://localhost:8080 in browser
```
