import json
import re
from typing import Dict, List, Optional

from llama_index.core import Document
from unstructured.documents.elements import Element

DOC_ID_SANITIZE_PATTERN = re.compile(r"[^a-zA-Z0-9_\-=]")


def sanitize_doc_id(doc_id: str) -> str:
    """
    Sanitize the doc_id to ensure it only contains allowed characters.
    Allowed characters: letters, digits, underscore (_), dash (-), or equal sign (=).
    """
    return DOC_ID_SANITIZE_PATTERN.sub("_", doc_id)


def to_documents(
    elements: List[Element],
    filename: str,
    extra_info: Optional[Dict] = None,
    deterministic_ids: bool = True,
) -> List[Document]:
    docs = []
    for i, node in enumerate(elements):
        if not hasattr(node, "metadata"):
            continue

        fields_to_skip = [
            "_known_field_names",  # does not serialize
            "DEBUG_FIELD_NAMES",  # does not serialize
            "coordinates",  # does not serialize
            "parent_id",  # might cause interference
            "orig_elements",  # does not serialize
        ]
        metadata = {}
        for field, val in vars(node.metadata).items():
            if field in fields_to_skip:
                continue

            try:
                metadata[field] = (
                    json.dumps(val)
                    if not isinstance(val, (str, int, float, type(None)))
                    else val
                )
            except (TypeError, OverflowError):
                continue

        if extra_info is not None:
            metadata.update(extra_info)

        metadata["filename"] = filename

        doc_kwargs = {
            "text": node.text,
            "extra_info": metadata,
        }

        if deterministic_ids:
            doc_kwargs["doc_id"] = sanitize_doc_id(f"{filename}_part_{i}")

        docs.append(Document(**doc_kwargs))

    return docs
