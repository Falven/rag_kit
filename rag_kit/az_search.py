import re


def sanitize_azure_search_index_name(input_string: str) -> str:
    """
    Sanitizes a given string to a valid Azure Search index name.

    The requirements for a valid Azure Search index name are:
    - Length: 2 to 128 characters
    - Allowed characters: Lowercase letters, numbers, dashes (-), and underscores (_)
    - First character must be a letter or number
    - No consecutive dashes or underscores
    """
    input_string = input_string.lower()

    valid_chars = re.sub(r"[^a-z0-9-_]", "", input_string)

    if not valid_chars[0].isalnum():
        valid_chars = "d" + valid_chars

    valid_chars = re.sub(r"[-_]{2,}", "-", valid_chars)

    if len(valid_chars) > 128:
        valid_chars = valid_chars[:128]

    if len(valid_chars) < 2:
        valid_chars = valid_chars + "a" * (2 - len(valid_chars))

    return valid_chars
