def test_import():
    import MEXsrfdcrPy as mex
    assert hasattr(mex, "__version__")
    assert callable(mex.hello)
    assert isinstance(mex.about(), str)
