import re

from bs4 import BeautifulSoup


def strip(s: str | None) -> str | None:
    if s is None:
        return None
    return s.strip()


def remove_html_tags(text: str) -> str:
    """Remove HTML tags from a string."""
    if "</span>" not in text:
        duplicated_spaces = re.compile(r"\s+")
        return duplicated_spaces.sub(" ", text).strip()

    soup = BeautifulSoup(text, "html.parser")
    lines = []
    for span in soup.select('span'):
        speaker = span.get('data-speaker')
        text = span.get_text(' ', strip=True)
        if speaker:
            lines.append(f'{speaker}: {text}')
        else:
            lines.append(text)
    _context = '\n'.join(lines)

    return _context


def has_placeholder(text: str, placeholder: str = "{CHAR}") -> bool:
    return placeholder in text


def fill_placeholders(template: str, placeholder: str, value: str) -> str:
    # try to replace this placeholder in the template
    if placeholder in template:
        return template.replace(placeholder, value)
    # otherwise return the original
    return template
