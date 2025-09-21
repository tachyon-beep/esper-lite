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

    def __init__(self, *, ttl_seconds: float = 300.0, clock: Callable[[], float] | None = None) -> None:
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

    def cleanup(self) -> None:
        now = self._clock()
        keys_to_remove = [key for key, (expiry, _) in self._store.items() if expiry < now]
        for key in keys_to_remove:
            del self._store[key]
            self._evictions += 1

    def stats(self) -> CacheStats:
        total = self._hits + self._misses
        hit_rate = (self._hits / total) if total else 0.0
        return CacheStats(size=len(self._store), hit_rate=hit_rate, evictions=self._evictions)

    def delete(self, key: str) -> None:
        self._store.pop(key, None)


@dataclass(slots=True)
class KasminaMemoryManager:
    """Aggregate caches used by Kasmina for governance."""

    kernel_cache: TTLMemoryCache[Any] = field(default_factory=TTLMemoryCache)
    telemetry_cache: TTLMemoryCache[Any] = field(default_factory=TTLMemoryCache)

    def cleanup(self) -> None:
        self.kernel_cache.cleanup()
        self.telemetry_cache.cleanup()


__all__ = ["KasminaMemoryManager", "TTLMemoryCache", "CacheStats"]
