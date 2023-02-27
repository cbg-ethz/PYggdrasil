from types import ModuleType

def test_install():
    import pyggdrasil as ygg

    assert isinstance(ygg, ModuleType)

