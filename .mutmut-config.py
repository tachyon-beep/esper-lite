"""Mutmut configuration for mutation testing.

Usage:
    # Test Tamiyo (default):
    uv run mutmut run

    # Test Kasmina:
    uv run mutmut run --paths-to-mutate=src/esper/kasmina/

    # Test specific file:
    uv run mutmut run --paths-to-mutate=src/esper/tamiyo/heuristic.py

    # View results:
    uv run mutmut results
    uv run mutmut show <id>
"""


def pre_mutation(context):
    """Filter mutations to focus on business logic, skip boilerplate."""
    filename = context.filename

    # Skip test files (shouldn't be in paths-to-mutate anyway)
    if "/tests/" in filename or filename.startswith("tests/"):
        context.skip = True
        return

    # Skip __init__.py files (usually just exports)
    if filename.endswith("__init__.py"):
        context.skip = True
        return

    # Skip type stubs
    if filename.endswith(".pyi"):
        context.skip = True
        return


def pre_mutation_ast(context):
    """Skip mutations in specific code patterns."""
    # Skip mutations in logging statements
    if context.current_source_line.strip().startswith("logger."):
        context.skip = True
        return

    # Skip mutations in docstrings
    if '"""' in context.current_source_line or "'''" in context.current_source_line:
        context.skip = True
        return
