import pytest
from icu_mm.data.inspection import parse_paths


def test_safe_path_column_shapes():
    assert parse_paths("['a.jpg', 'b.jpg']") == ['a.jpg', 'b.jpg']
    assert parse_paths('a.jpg') == ['a.jpg']
    assert parse_paths(float('nan')) == []
    with pytest.raises((ValueError, SyntaxError)):
        parse_paths("[__import__('os').getcwd()]")
    with pytest.raises(ValueError):
        parse_paths('[42]')
