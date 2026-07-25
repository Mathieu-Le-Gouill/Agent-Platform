import pytest
from pydantic import ValidationError

from agent_platform.core.schemas.vector import SparseVector


class TestSparseVector:
    def test_matching_lengths_constructs(self):
        vec = SparseVector(indices=[0, 2, 5], values=[0.1, 0.2, 0.3])
        assert vec.indices == [0, 2, 5]
        assert vec.values == [0.1, 0.2, 0.3]

    def test_empty_vector_is_valid(self):
        vec = SparseVector(indices=[], values=[])
        assert vec.indices == []

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValidationError, match="length mismatch"):
            SparseVector(indices=[0, 1], values=[0.1])

    def test_is_frozen(self):
        vec = SparseVector(indices=[0], values=[0.1])
        with pytest.raises(ValidationError):
            vec.indices = [1]
