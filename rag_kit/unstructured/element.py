import json
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
        element = self._element

        if deterministic_ids and sequence_number is None:
            raise ValueError(
                "If deterministic_ids is True, sequence_number must be provided."
            )

        doc_kwargs = {"text": element.text}

        metadata = element.metadata.to_dict()

        if extra_info:
            metadata.update(extra_info)

        for key, value in metadata.items():
            if not isinstance(value, (str, int, float, type(None))):
                metadata[key] = json.dumps(value)

        doc_kwargs["extra_info"] = metadata

        if deterministic_ids:
            doc_kwargs["doc_id"] = element.id_to_hash(sequence_number)

        return Document(**doc_kwargs)
