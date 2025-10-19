import importlib

def test_import_package():
    pkg = importlib.import_module("MEXsrfdcrPy")
    assert hasattr(pkg, "__version__")

def test_hello_about():
    pkg = importlib.import_module("MEXsrfdcrPy")
    # Estas funciones vienen del __init__.py de tu esqueleto
    assert "Hello" in pkg.hello("Mexico")
    assert "Spatial RF" in pkg.about()
