// src/esper/karn/overwatch/web/src/test-setup.ts
// Test environment setup for Vitest

// Mock ResizeObserver for JSDOM environment
global.ResizeObserver = class ResizeObserver {
  observe() {}
  unobserve() {}
  disconnect() {}
}
