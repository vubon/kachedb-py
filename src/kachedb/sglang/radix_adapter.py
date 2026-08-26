"""
Radix tree adapter and variable-length token hashing for SGLang integration.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RadixNodeDescriptor:
    """Metadata descriptor for an evicted SGLang Radix tree node stored in KacheDB."""

    node_id: int
    parent_hash: int
    node_hash: int
    token_ids: list[int]
    num_tokens: int
    layer_offsets: list[int] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class KacheDBRadixAdapter:
    """Fast rolling Radix tree prefix index for SGLang KV-cache subtrees.

    Unlike fixed PagedAttention block chunks, SGLang tree nodes hold
    variable-length token slices. This adapter manages chained 64-bit Blake2b
    hashes for arbitrary token sequences and tracks node descriptors.
    """

    def __init__(self) -> None:
        # In-memory mapping from node_hash to RadixNodeDescriptor
        self._nodes: dict[int, RadixNodeDescriptor] = {}
        # Root hash is 0
        self._root_hash: int = 0

    @staticmethod
    def compute_node_hash(token_ids: list[int], parent_hash: int = 0) -> int:
        """Compute a deterministic 64-bit hash for a variable-length token slice.

        Parameters
        ----------
        token_ids : list[int]
            List of integer token IDs stored in the Radix tree node.
        parent_hash : int
            64-bit hash of the parent tree node (default: 0 for root).

        Returns
        -------
        int
            Unsigned 64-bit integer hash.
        """
        buf = bytearray(12 + len(token_ids) * 4)
        # 8 bytes parent_hash + 4 bytes token_count
        struct.pack_into("<QI", buf, 0, parent_hash, len(token_ids))
        offset = 12
        for tid in token_ids:
            struct.pack_into("<I", buf, offset, tid)
            offset += 4

        digest = hashlib.blake2b(buf, digest_size=8).digest()
        val: tuple[int, ...] = struct.unpack("<Q", digest)
        return int(val[0])

    def register_node(
        self,
        node_id: int,
        token_ids: list[int],
        parent_hash: int,
        layer_offsets: list[int],
        metadata: dict[str, Any] | None = None,
    ) -> RadixNodeDescriptor:
        """Register an evicted Radix node in the local descriptor index.

        Returns
        -------
        RadixNodeDescriptor
            The created and indexed node descriptor.
        """
        node_hash = self.compute_node_hash(token_ids, parent_hash=parent_hash)
        desc = RadixNodeDescriptor(
            node_id=node_id,
            parent_hash=parent_hash,
            node_hash=node_hash,
            token_ids=token_ids,
            num_tokens=len(token_ids),
            layer_offsets=layer_offsets,
            metadata=metadata or {},
        )
        self._nodes[node_hash] = desc
        return desc

    def lookup_node(self, node_hash: int) -> RadixNodeDescriptor | None:
        """Look up a Radix node descriptor by its 64-bit hash."""
        return self._nodes.get(node_hash)

    def find_matching_nodes(
        self, prompt_tokens: list[int], max_tokens: int | None = None
    ) -> tuple[int, list[RadixNodeDescriptor]]:
        """Find matching cached Radix tree nodes for a prompt token sequence.

        Traverses matching nodes starting from root.

        Returns
        -------
        tuple[int, list[RadixNodeDescriptor]]
            (matched_tokens_count, list_of_matching_descriptors)
        """
        matched_tokens = 0
        matched_nodes: list[RadixNodeDescriptor] = []
        current_parent = 0
        token_cursor = 0
        limit = max_tokens if max_tokens is not None else len(prompt_tokens)

        while token_cursor < limit:
            best_node: RadixNodeDescriptor | None = None
            # Check variable slice candidates
            for _node_hash, desc in self._nodes.items():
                if desc.parent_hash == current_parent:
                    n_len = desc.num_tokens
                    if token_cursor + n_len <= len(prompt_tokens):
                        candidate_tokens = prompt_tokens[token_cursor : token_cursor + n_len]
                        if candidate_tokens == desc.token_ids:
                            best_node = desc
                            break

            if best_node is None:
                break

            matched_nodes.append(best_node)
            matched_tokens += best_node.num_tokens
            token_cursor += best_node.num_tokens
            current_parent = best_node.node_hash

        return matched_tokens, matched_nodes
