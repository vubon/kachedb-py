"""
Prefix caching and token chunk hashing for KacheDB vLLM integration.
"""

from __future__ import annotations

import hashlib
import struct
from typing import Dict, List, Optional, Tuple


class KacheDBPrefixCache:
    """Fast rolling token-chunk prefix index for LLM KV-cache blocks.

    Divides token sequences into fixed-size PagedAttention block chunks
    (default: 16 or 32 tokens) and computes deterministic 64-bit sequence hashes.
    """

    def __init__(self, block_size: int = 16):
        self.block_size = block_size
        # In-memory mapping from prefix_hash to (block_id, num_tokens, metadata)
        self._index: Dict[int, Tuple[int, int, dict]] = {}

    def compute_block_hash(self, token_ids: List[int], parent_hash: int = 0) -> int:
        """Compute a deterministic 64-bit hash for a token block chained to its parent.

        Parameters
        ----------
        token_ids : List[int]
            List of integer token IDs for the block (length <= block_size).
        parent_hash : int
            Hash of the preceding token block in the sequence.

        Returns
        -------
        int
            Unsigned 64-bit integer hash.
        """
        # Pack parent hash and token IDs into binary buffer
        buf = bytearray(8 + len(token_ids) * 4)
        struct.pack_into("<Q", buf, 0, parent_hash)
        offset = 8
        for tid in token_ids:
            struct.pack_into("<I", buf, offset, tid)
            offset += 4

        # Blake3 or fast 64-bit MD5 truncated digest
        digest = hashlib.blake2b(buf, digest_size=8).digest()
        return struct.unpack("<Q", digest)[0]

    def compute_sequence_hashes(self, prompt_tokens: List[int]) -> List[int]:
        """Compute chained block hashes for an entire prompt token sequence.

        Parameters
        ----------
        prompt_tokens : List[int]
            Full list of prompt token IDs.

        Returns
        -------
        List[int]
            List of 64-bit block hashes, one per block_size chunk.
        """
        hashes: List[int] = []
        current_hash = 0
        num_blocks = len(prompt_tokens) // self.block_size

        for b in range(num_blocks):
            chunk = prompt_tokens[b * self.block_size : (b + 1) * self.block_size]
            current_hash = self.compute_block_hash(chunk, parent_hash=current_hash)
            hashes.append(current_hash)

        return hashes

    def register_block(
        self,
        prefix_hash: int,
        block_id: int,
        num_tokens: int,
        metadata: Optional[dict] = None,
    ) -> None:
        """Register a cached block in the local prefix index."""
        self._index[prefix_hash] = (block_id, num_tokens, metadata or {})

    def lookup_block(self, prefix_hash: int) -> Optional[Tuple[int, int, dict]]:
        """Look up a block in the prefix index by its chained hash."""
        return self._index.get(prefix_hash)

    def find_longest_prefix(
        self, prompt_tokens: List[int]
    ) -> Tuple[int, List[Tuple[int, int, dict]]]:
        """Find the longest matching prefix for a prompt.

        Returns
        -------
        Tuple[int, List[Tuple[int, int, dict]]]
            (matched_tokens_count, list_of_matched_block_tuples)
        """
        hashes = self.compute_sequence_hashes(prompt_tokens)
        matched_blocks = []
        matched_tokens = 0

        for h in hashes:
            entry = self.lookup_block(h)
            if entry is None:
                break
            matched_blocks.append(entry)
            matched_tokens += self.block_size

        return matched_tokens, matched_blocks
