import MEXsrfdcrPy as mex

def test_hello():
    assert mex.hello("Mexico").startswith("Hello,")

def test_about():
    assert "Spatial RF" in mex.about()
