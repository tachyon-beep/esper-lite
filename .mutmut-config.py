"""Mutmut configuration for Tamiyo mutation testing."""


def pre_mutation(context):
    """Filter mutations to only Tamiyo modules."""
    # Only mutate files in src/esper/tamiyo/
    if not context.filename.startswith("src/esper/tamiyo/"):
        context.skip = True
