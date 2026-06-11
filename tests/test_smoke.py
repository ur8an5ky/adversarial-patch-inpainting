from src import interfaces


def test_interfaces_exist():
    assert callable(interfaces.detect_mask)
    assert callable(interfaces.inpaint)
    assert callable(interfaces.evaluate)