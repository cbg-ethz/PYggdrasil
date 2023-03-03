"""Check if basic functionalities of the public API are available."""
from types import ModuleType


def test_install() -> None:
    """Global module import."""
    import pyggdrasil as ygg

    assert isinstance(ygg, ModuleType)
