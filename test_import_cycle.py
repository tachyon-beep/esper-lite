"""Test to verify the import cycle and track what gets loaded."""
import sys

# Clear any previously loaded esper modules
esper_modules = [k for k in sys.modules.keys() if k.startswith('esper.')]
for mod in esper_modules:
    del sys.modules[mod]

print("=== Before any imports ===")
print(f"esper.* modules loaded: {len([k for k in sys.modules if k.startswith('esper.')])}")

print("\n=== Importing esper.tolaria ===")
import esper.tolaria  # noqa: E402 - deliberate import after sys.modules clear
print(f"esper.* modules loaded: {len([k for k in sys.modules if k.startswith('esper.')])}")

# Check if heavy modules got loaded
heavy_modules = [
    'esper.simic.training.vectorized',
    'torch',
    'esper.nissa',
]

for mod in heavy_modules:
    if mod in sys.modules:
        print(f"  ⚠️  {mod} loaded (should be lazy!)")
    else:
        print(f"  ✓  {mod} NOT loaded (good)")

print("\n=== Importing esper.runtime ===")
import esper.runtime  # noqa: E402, F401 - deliberate import after sys.modules clear
print(f"esper.* modules loaded: {len([k for k in sys.modules if k.startswith('esper.')])}")

for mod in heavy_modules:
    if mod in sys.modules:
        print(f"  ⚠️  {mod} loaded (should be lazy!)")
    else:
        print(f"  ✓  {mod} NOT loaded (good)")

# Show all loaded esper modules
print("\n=== All loaded esper.* modules ===")
esper_mods = sorted([k for k in sys.modules if k.startswith('esper.')])
for mod in esper_mods:
    print(f"  - {mod}")
