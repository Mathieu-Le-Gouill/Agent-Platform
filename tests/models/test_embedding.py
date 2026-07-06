import math
from uuid import uuid4

import pytest

from agent_platform.models.embedding import Embedding


class TestEmbedding:
    def test_from_list(self):
        emb = Embedding.from_list([0.1, 0.2, 0.3])
        assert emb.vector == (0.1, 0.2, 0.3)

    def test_to_list(self):
        emb = Embedding.from_list([0.1, 0.2, 0.3])
        assert emb.to_list() == [0.1, 0.2, 0.3]

    def test_to_list_returns_copy(self):
        emb = Embedding.from_list([0.1, 0.2])
        lst = emb.to_list()
        lst.append(0.3)
        assert emb.dimensions == 2

    def test_dimensions(self):
        emb = Embedding.from_list([0.1, 0.2, 0.3, 0.4])
        assert emb.dimensions == 4

    def test_norm(self):
        emb = Embedding.from_list([3.0, 4.0])
        assert emb.norm == 5.0

    def test_norm_zero_vector(self):
        emb = Embedding.from_list([0.0, 0.0, 0.0])
        assert emb.norm == 0.0

    def test_norm_unit_vector(self):
        val = 1.0 / math.sqrt(3.0)
        emb = Embedding.from_list([val, val, val])
        assert math.isclose(emb.norm, 1.0, rel_tol=1e-6)

    def test_is_normalized_true(self):
        val = 1.0 / math.sqrt(2.0)
        emb = Embedding.from_list([val, val])
        assert emb.is_normalized() is True

    def test_is_normalized_false(self):
        emb = Embedding.from_list([3.0, 4.0])
        assert emb.is_normalized() is False

    def test_is_normalized_with_custom_tol(self):
        emb = Embedding.from_list([1.0, 0.001])
        assert emb.is_normalized(tol=1e-2) is True

    def test_empty_vector_raises(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            Embedding(vector=())

    def test_from_list_empty_raises(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            Embedding.from_list([])

    def test_frozen(self):
        emb = Embedding.from_list([1.0, 2.0])
        with pytest.raises(ValueError):
            emb.vector = (3.0, 4.0)

    def test_custom_id(self):
        uid = uuid4()
        emb = Embedding.from_list([1.0], id=uid)
        assert emb.id == uid

    def test_custom_model(self):
        emb = Embedding.from_list([1.0], model="text-embedding-3-small")
        assert emb.model == "text-embedding-3-small"
