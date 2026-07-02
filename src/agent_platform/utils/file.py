from pathlib import Path

def get_file_extension(file_path: str, include_dot: bool = False) -> str:
    """
    Extract the file extension from a file path.

    Args:
        file_path (str): Path to the file.
        include_dot (bool): If True, returns '.txt' instead of 'txt'.

    Returns:
        str: The file extension or an empty string if none exists.
    """
    ext = Path(file_path).suffix
    if not include_dot:
        ext = ext.lstrip(".")
    return ext


def get_document_title(file_path: str) -> str:
    """
    Extract the document title from a file path.
    (Filename without extension)

    Args:
        file_path (str): Path to the file.

    Returns:
        str: File name without extension.
    """
    return Path(file_path).stem