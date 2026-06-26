from uuid import UUID, uuid4


def parse_uuid(val) -> UUID:
    try:
        return UUID(str(val))
    except (TypeError, ValueError):
        return uuid4()