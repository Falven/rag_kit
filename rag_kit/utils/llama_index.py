import json
from typing import Dict, List, Optional

from llama_index.core import Document
from unstructured.documents.elements import Element


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

        metadata = {}
        for field, val in vars(node.metadata).items():
            if field == "_known_field_names":
                continue
            # removing coordinates because it does not serialize
            # and dont want to bother with it
            if field == "coordinates":
                continue
            # removing bc it might cause interference
            if field == "parent_id":
                continue

            metadata[field] = (
                json.dumps(val)
                if not isinstance(val, (str, int, float, type(None)))
                else val
            )

        if extra_info is not None:
            metadata.update(extra_info)

        metadata["filename"] = filename

        doc_kwargs = {
            "text": node.text,
            "extra_info": metadata,
        }

        if deterministic_ids:
            doc_kwargs["id_"] = f"{filename!s}_part_{i}"

        docs.append(Document(**doc_kwargs))

    return docs
