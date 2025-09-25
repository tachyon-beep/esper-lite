"""Kasmina memory governance utilities."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Generic, MutableMapping, TypeVar

T = TypeVar("T")


@dataclass(slots=True)
class CacheStats:
    size: int
    hit_rate: float
    evictions: int


class TTLMemoryCache(Generic[T]):
    """Simple TTL cache with eviction statistics."""

    def __init__(
        self, *, ttl_seconds: float = 300.0, clock: Callable[[], float] | None = None
    ) -> None:
        self._ttl = ttl_seconds
        self._clock = clock or time.monotonic
        self._store: MutableMapping[str, tuple[float, T]] = {}
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def set(self, key: str, value: T) -> None:
        self._store[key] = (self._clock() + self._ttl, value)

    def get(self, key: str) -> T | None:
        expiry_value = self._store.get(key)
        if not expiry_value:
            self._misses += 1
            return None
        expiry, value = expiry_value
        if expiry < self._clock():
            del self._store[key]
            self._evictions += 1
            self._misses += 1
            return None
        self._hits += 1
        return value

    def cleanup(self) -> int:
        now = self._clock()
        keys_to_remove = [key for key, (expiry, _) in self._store.items() if expiry < now]
        for key in keys_to_remove:
            del self._store[key]
            self._evictions += 1

        return len(keys_to_remove)

    def stats(self) -> CacheStats:
        total = self._hits + self._misses
        hit_rate = (self._hits / total) if total else 0.0
        return CacheStats(size=len(self._store), hit_rate=hit_rate, evictions=self._evictions)

    def delete(self, key: str) -> None:
        self._store.pop(key, None)

    def clear(self) -> None:
        self._store.clear()
        self._hits = 0
        self._misses = 0
        self._evictions = 0


@dataclass(slots=True)
class KasminaMemoryManager:
    """Aggregate caches used by Kasmina for governance."""

    kernel_cache: TTLMemoryCache[Any] = field(default_factory=TTLMemoryCache)
    telemetry_cache: TTLMemoryCache[Any] = field(default_factory=TTLMemoryCache)
    gc_frequency: int = 10
    _last_gc_epoch: int = 0
    _gc_counter: int = 0

    def cleanup(self) -> None:
        self.kernel_cache.cleanup()
        self.telemetry_cache.cleanup()

    def periodic_gc(self, epoch: int) -> dict[str, int]:
        """Run periodic garbage collection based on the configured frequency."""

        if self.gc_frequency <= 0:
            return {"gc_skipped": 1, "reason": "disabled"}
        if epoch < self._last_gc_epoch:
            self._last_gc_epoch = epoch
        if (epoch - self._last_gc_epoch) < self.gc_frequency:
            return {"gc_skipped": 1, "reason": "frequency_not_met"}

        kernel_expired = self.kernel_cache.cleanup()
        telemetry_expired = self.telemetry_cache.cleanup()
        try:
            import gc

            python_collected = gc.collect()
        except Exception:  # pragma: no cover - optional GC hook
            python_collected = 0

        self._gc_counter += 1
        self._last_gc_epoch = epoch
        return {
            "kernel_cache_expired": kernel_expired,
            "telemetry_cache_expired": telemetry_expired,
            "python_gc_collected": python_collected,
            "gc_counter": self._gc_counter,
            "gc_epoch": epoch,
        }

    def emergency_cleanup(self, *, include_teacher: bool = False) -> dict[str, int]:
        """Aggressively clear caches and trigger Python garbage collection."""

        kernel_stats = self.kernel_cache.stats()
        telemetry_stats = self.telemetry_cache.stats()
        preserved_teacher = None
        if not include_teacher:
            preserved_teacher = self.kernel_cache.get("teacher")
        self.kernel_cache.clear()
        if preserved_teacher is not None:
            self.kernel_cache.set("teacher", preserved_teacher)
        self.telemetry_cache.clear()
        try:
            import gc

            python_collected = gc.collect()
        except Exception:  # pragma: no cover
            python_collected = 0
        return {
            "kernel_cache_cleared": kernel_stats.size,
            "telemetry_cache_cleared": telemetry_stats.size,
            "python_gc_collected": python_collected,
        }


__all__ = ["KasminaMemoryManager", "TTLMemoryCache", "CacheStats"]
