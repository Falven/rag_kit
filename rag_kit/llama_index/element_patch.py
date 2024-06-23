import json
from typing import Optional

from llama_index.core import Document
from unstructured.documents.elements import Element


def to_document(
    self: Element,
    sequence_number: Optional[int] = None,
    extra_info: Optional[dict] = None,
    deterministic_ids: bool = False,
) -> Document:
    """
    Convert an Element to a Document.

    Args:
        sequence_number: The sequence number of the element, used for deterministic ID generation.
        extra_info: Additional metadata to include in the document.
        deterministic_ids: If True, the document ID will be deterministic based on the element's text and other properties.
            If False, the document ID will be a random UUID.

    Returns:
        Document: The converted Document.
    """
    if deterministic_ids and sequence_number is None:
        raise ValueError(
            "If deterministic_ids is True, sequence_number must be provided."
        )

    fields_to_skip = [
        "_known_field_names",  # does not serialize
        "DEBUG_FIELD_NAMES",  # does not serialize
        "coordinates",  # does not serialize
        "parent_id",  # might cause interference
        "orig_elements",  # does not serialize
    ]
    transient_fields = (str, int, float, type(None))
    metadata = {}
    if self.metadata:
        for field, val in vars(self.metadata).items():
            if field in fields_to_skip:
                continue

            try:
                metadata[field] = (
                    json.dumps(val) if not isinstance(val, transient_fields) else val
                )
            except (TypeError, OverflowError):
                continue

        if extra_info is not None:
            metadata.update(extra_info)

    doc_kwargs = {
        "text": self.text,
        "extra_info": metadata,
    }

    if deterministic_ids:
        doc_kwargs["doc_id"] = self.id_to_hash(sequence_number)

    return Document(**doc_kwargs)


Element.to_document = to_document
