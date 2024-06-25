import json
from pydoc import doc
from typing import Optional

from llama_index.core import Document
from unstructured.documents.elements import Element


class ElementComposer:
    def __init__(self, element: Element):
        self._element = element

    def __getattr__(self, name):
        """
        Delegate attribute access to the wrapped Element instance.
        """
        return getattr(self._element, name)


def to_document(
    self,
    sequence_number: Optional[int] = None,
    extra_info: Optional[dict] = None,
    deterministic_ids: bool = False,
    excluded_metadata_keys: Optional[list] = None,
    document_kwargs: Optional[dict] = None,
) -> Document:
    """
    Convert an Element to a Document.

    Args:
        sequence_number (Optional[int]): The sequence number of the element, used for deterministic ID generation.
                                          Required if `deterministic_ids` is True.
        extra_info (Optional[dict]): Additional metadata to include in the document.
        deterministic_ids (bool): If True, the document ID will be deterministic based on the element's text and other properties.
                                  If False, the document ID will be a random UUID.
        excluded_metadata (Optional[list]): List of metadata keys to exclude from the document's metadata.
                                            Defaults to ["orig_elements"].
        document_kwargs (Optional[dict]): Additional keyword arguments to pass to the Document constructor.

    Returns:
        Document: The converted Document.

    Raises:
        ValueError: If `deterministic_ids` is True and `sequence_number` is not provided.
    """
    if deterministic_ids and sequence_number is None:
        raise ValueError(
            "If deterministic_ids is True, sequence_number must be provided."
        )

    kwargs = {"text": self._element.text}
    if document_kwargs:
        kwargs.update(document_kwargs)

    metadata = {**extra_info} if extra_info else {}
    excluded_metadata_keys = excluded_metadata_keys or ["orig_elements"]

    for key, value in self._element.metadata.to_dict().items():
        if key not in excluded_metadata_keys:
            metadata[key] = (
                value
                if isinstance(value, (str, int, float, type(None)))
                else json.dumps(value)
            )

    kwargs["extra_info"] = metadata

    if deterministic_ids:
        kwargs["doc_id"] = self._element.id_to_hash(sequence_number)

    return Document(**kwargs)
