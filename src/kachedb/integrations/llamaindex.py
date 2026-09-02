"""
LlamaIndex Storage Integration for KacheDB.

Provides high-throughput in-memory Key-Value and Index storage backends for LlamaIndex.
"""

from __future__ import annotations

import json
from typing import Any

from ..client import KacheClient


class KacheDBKVStore:
    """KacheDB in-memory Key-Value store adapter for LlamaIndex.

    Parameters
    ----------
    client : KacheClient | None
        Active KacheDB client.
    namespace : str
        Default namespace prefix.
    """

    def __init__(
        self,
        client: KacheClient | None = None,
        namespace: str = "llamaindex:kv",
    ) -> None:
        self.client = client or KacheClient()
        self.namespace = namespace

    def _format_key(self, key: str, collection: str) -> str:
        return f"{self.namespace}:{collection}:{key}"

    def put(self, key: str, val: dict[str, Any], collection: str = "data") -> None:
        """Put a key-value pair in the store."""
        full_key = self._format_key(key, collection)
        self.client.set(full_key, json.dumps(val))

    def put_all(self, kv_pairs: list[tuple[str, dict[str, Any]]], collection: str = "data") -> None:
        """Put multiple key-value pairs in the store."""
        pipe = self.client.pipeline()
        for key, val in kv_pairs:
            full_key = self._format_key(key, collection)
            pipe.set(full_key, json.dumps(val))
        pipe.execute()

    def get(self, key: str, collection: str = "data") -> dict[str, Any] | None:
        """Get a value from the store."""
        full_key = self._format_key(key, collection)
        val = self.client.get(full_key)
        if val is None:
            return None
        val_str = val.decode("utf-8") if isinstance(val, bytes) else str(val)
        try:
            res: dict[str, Any] = json.loads(val_str)
            return res
        except Exception:
            return None

    def delete(self, key: str, collection: str = "data") -> bool:
        """Delete a key from the store."""
        full_key = self._format_key(key, collection)
        return self.client.delete(full_key) == 1


class KacheDBIndexStore:
    """KacheDB in-memory Index store adapter for LlamaIndex."""

    def __init__(
        self,
        kvstore: KacheDBKVStore | None = None,
        namespace: str = "llamaindex:index",
    ) -> None:
        self.kvstore = kvstore or KacheDBKVStore(namespace=namespace)

    def put(self, index_id: str, index_struct: dict[str, Any]) -> None:
        """Store index structure."""
        self.kvstore.put(index_id, index_struct, collection="index")

    def get(self, index_id: str) -> dict[str, Any] | None:
        """Retrieve index structure."""
        return self.kvstore.get(index_id, collection="index")

    def delete(self, index_id: str) -> bool:
        """Delete index structure."""
        return self.kvstore.delete(index_id, collection="index")
