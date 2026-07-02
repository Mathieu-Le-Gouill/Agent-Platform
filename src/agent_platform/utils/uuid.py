from uuid import UUID, uuid4


def parse_uuid(val) -> UUID:
    try:
        return UUID(str(val))
    except (TypeError, ValueError):
        return uuid4()
    

def uuid_to_str(val: UUID) -> str:
    try:
        return str(val)
    except (TypeError, ValueError):
        return ""
    

def new_uuid() -> UUID:
    return uuid4()