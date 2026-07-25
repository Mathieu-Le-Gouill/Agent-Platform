from datetime import datetime
from uuid import UUID

from agent_platform.core.base import Entity, Timestamped


class TestEntity:
    def test_id_defaults_to_a_unique_uuid(self):
        first = Entity()
        second = Entity()

        assert isinstance(first.id, UUID)
        assert first.id != second.id

    def test_id_can_be_set_explicitly(self):
        fixed_id = UUID("12345678-1234-5678-1234-567812345678")

        entity = Entity(id=fixed_id)

        assert entity.id == fixed_id


class TestTimestamped:
    def test_created_at_defaults_to_now(self):
        before = datetime.now()
        stamped = Timestamped()
        after = datetime.now()

        assert before <= stamped.created_at <= after

    def test_modified_at_defaults_to_none(self):
        stamped = Timestamped()

        assert stamped.modified_at is None

    def test_modified_at_can_be_set_explicitly(self):
        modified = datetime(2026, 1, 1)

        stamped = Timestamped(modified_at=modified)

        assert stamped.modified_at == modified
