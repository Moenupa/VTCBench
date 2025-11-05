import base64
from io import BytesIO

from PIL import Image as PILImage

from ..defaults import PIL_SAVE_FORMAT, PIL_SAVE_KWARGS


class ImageTextPayload:
    # for the "content" field in chat messages, but only for image data
    # this should be formatted as
    # "content": [{"type": "image", "image": {"data_url": "<data‑URL string>"}}]
    def __init__(self):
        self.payloads: list[dict] = []

    def add_text(self, text: str):
        self.payloads.append({"type": "text", "text": text.strip()})

    def add_image_adaptive(self, image: bytes | str | PILImage.Image, ext: str = None):
        self.payloads.append(
            {
                "type": "image_url",
                "image_url": {"url": adaptive_image_to_data_url(image, ext)},
            }
        )

    def to_message_content(self) -> list[dict]:
        for payload in self.payloads:
            # check that it is in correct format
            assert isinstance(payload, dict)
            assert "type" in payload and payload["type"] in payload

        return self.payloads


def image_path_to_data_url(image_path: str, ext: str = None) -> str:
    """Convert an image file to a data URL.

    Args:
        image_path (str): The path to the image file.
        ext (optional, str): The image file extension (e.g., "jpeg", "png"). If None, it will be inferred from the file name.

    Returns:
        str: data‑URL string, e.g. "data:<mime>;base64,<payload>"
    """
    if ext is None:
        ext = image_path.split(".")[-1]

    with open(image_path, "rb") as image_file:
        return image_bytes_to_data_url(image_file.read(), ext)


def image_object_to_data_url(
    image_object: PILImage.Image, ext: str = PIL_SAVE_FORMAT
) -> str:
    """Convert an image object to a data URL.

    Args:
        image_object (PIL.Image.Image): The pillow image object.
        ext (str): The image file extension (e.g., "jpeg", "png").

    Returns:
        str: data‑URL string, e.g. "data:<mime>;base64,<payload>"
    """
    # special case for jpg -> pillow's jpeg
    if ext == "jpg":
        ext = "jpeg"

    kwargs = PIL_SAVE_KWARGS.get(ext.lower(), {})
    with BytesIO() as buffered:
        image_object.save(buffered, format=ext.upper(), **kwargs)
        bs = buffered.getvalue()
    return image_bytes_to_data_url(bs, ext)


def image_bytes_to_data_url(image_bytes: bytes, ext: str) -> str:
    """Convert image bytes to a data URL.

    Args:
        image_bytes (bytes): The image bytes.
        ext (str): The image file extension (e.g., "jpeg", "png").

    Returns:
        str: data‑URL string, e.g. "data:<mime>;base64,<payload>"
    """
    encoded_string = base64.b64encode(image_bytes).decode("utf-8")
    data_url = f"data:image/{ext.lower()};base64,{encoded_string}"
    return data_url


def adaptive_image_to_data_url(
    image: bytes | str | PILImage.Image, ext: str = None
) -> str:
    r"""Resize the image if larger than max_size and convert to data URL.

    Args:
        image (adaptive): Bytes, local path, or Pillow Image Object.
        max_size (int): The maximum size for width or height.

    Returns:
        str: data‑URL string, e.g. ``data:<mime>;base64,<payload>``
    """
    if isinstance(image, bytes):
        # ext must be provided, no guessing
        assert ext is not None
        return image_bytes_to_data_url(image, ext)
    elif isinstance(image, str):
        # ext optional, prefer to be infered from file ext
        return image_path_to_data_url(image, ext)
    elif isinstance(image, PILImage.Image):
        # no ext needed, default to jpeg
        return image_object_to_data_url(image)

    raise NotImplementedError(f"Unsupported image input {image} with ext {ext}.")
