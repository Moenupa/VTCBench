import re


def strip(s: str | None) -> str | None:
    if s is None:
        return None
    return s.strip()


def remove_html_tags(text: str) -> str:
    """Remove HTML tags from a string."""
    clean = re.compile("<.*?>")
    duplicated_spaces = re.compile(r"\s+")
    return re.sub(duplicated_spaces, " ", re.sub(clean, " ", text))


def has_placeholder(text: str, placeholder: str = "{CHAR}") -> bool:
    return placeholder in text


def fill_placeholders(template: str, placeholder: str, value: str) -> str:
    # try to replace this placeholder in the template
    if placeholder in template:
        return template.replace(placeholder, value)
    # otherwise return the original
    return template
