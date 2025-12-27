// src/esper/karn/overwatch/web/src/test-setup.ts
// Test environment setup for Vitest

// Mock ResizeObserver for JSDOM environment
// (JSDOM doesn't implement ResizeObserver natively)
globalThis.ResizeObserver = class ResizeObserver {
  observe() {}
  unobserve() {}
  disconnect() {}
}
