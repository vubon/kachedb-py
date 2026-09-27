"""
Hierarchical Markdown and code chunker for KacheDB context engine.

Partitions structured technical documents, source code, and design notes
into semantic chunks while preserving heading hierarchy, code fence integrity,
and parent document linkage.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Regex for Markdown ATX headings (# Heading)
HEADING_REGEX = re.compile(r"^(#{1,6})\s+(.+)$")
# Regex for fenced code block delimiters
CODE_FENCE_REGEX = re.compile(r"^```")


@dataclass(frozen=True)
class DocumentChunk:
    """Represents a discrete semantic chunk of a larger parent document.

    Attributes
    ----------
    chunk_id : str
        Unique identifier for the chunk, typically '{parent_key}:chunk_{index}'.
    content : str
        The textual content of the chunk, including hierarchical section context.
    section_title : str
        Hierarchical breadcrumb of headings leading to this chunk (e.g. 'Overview > Architecture').
    parent_key : str
        Primary key of the full document stored in the KV store.
    token_count : int
        Estimated token count (~4 characters per token).
    """

    chunk_id: str
    content: str
    section_title: str
    parent_key: str
    token_count: int


def estimate_tokens(text: str) -> int:
    """Estimate token count for standard transformer embedding models (~4 chars/token)."""
    if not text:
        return 0
    return max(1, len(text) // 4)


class MarkdownChunker:
    """Header-aware Markdown chunker that preserves document structure and hierarchy.

    Parameters
    ----------
    target_chunk_size : int
        Target chunk size in tokens (default: 350 tokens, ~1,400 characters).
    overlap : int
        Token overlap between consecutive chunks within the same section (default: 50 tokens).
    """

    def __init__(self, target_chunk_size: int = 350, overlap: int = 50) -> None:
        self.target_size = max(50, target_chunk_size)
        self.overlap = min(overlap, self.target_size // 2)

    def chunk_document(self, parent_key: str, markdown_text: str) -> list[DocumentChunk]:
        """Split a Markdown document into hierarchical chunks linked to parent_key.

        Parameters
        ----------
        parent_key : str
            Unique key identifying the parent document in KacheDB (e.g. 'doc:workspace:topic').
        markdown_text : str
            Full Markdown document text.

        Returns
        -------
        list[DocumentChunk]
            Ordered list of document chunks.
        """
        text = markdown_text.strip()
        if not text:
            return []

        # If document is already small enough, return as single chunk
        total_tokens = estimate_tokens(text)
        if total_tokens <= self.target_size:
            return [
                DocumentChunk(
                    chunk_id=f"{parent_key}:chunk_0",
                    content=text,
                    section_title="Root",
                    parent_key=parent_key,
                    token_count=total_tokens,
                )
            ]

        lines = text.splitlines()
        chunks: list[DocumentChunk] = []
        chunk_idx = 0

        # Stack of (level, title) tracking heading hierarchy
        heading_stack: list[tuple[int, str]] = []

        current_lines: list[str] = []
        in_code_block = False

        def get_current_breadcrumb() -> str:
            if not heading_stack:
                return "General"
            return " > ".join(title for _, title in heading_stack)

        def flush_current_chunk() -> None:
            nonlocal chunk_idx, current_lines
            content = "\n".join(current_lines).strip()
            if not content:
                current_lines = []
                return

            tokens = estimate_tokens(content)
            title = get_current_breadcrumb()

            chunks.append(
                DocumentChunk(
                    chunk_id=f"{parent_key}:chunk_{chunk_idx}",
                    content=content,
                    section_title=title,
                    parent_key=parent_key,
                    token_count=tokens,
                )
            )
            chunk_idx += 1

            # Keep overlap lines if possible
            if self.overlap > 0 and len(current_lines) > 1:
                overlap_lines: list[str] = []
                accum_tokens = 0
                for line in reversed(current_lines):
                    line_tokens = estimate_tokens(line)
                    if accum_tokens + line_tokens > self.overlap:
                        break
                    overlap_lines.insert(0, line)
                    accum_tokens += line_tokens
                current_lines = overlap_lines
            else:
                current_lines = []

        for line in lines:
            # Check for code fence toggles
            if CODE_FENCE_REGEX.match(line):
                in_code_block = not in_code_block

            # Headings outside code blocks trigger section boundary checks
            if not in_code_block:
                heading_match = HEADING_REGEX.match(line)
                if heading_match:
                    level = len(heading_match.group(1))
                    title = heading_match.group(2).strip()

                    # Pop headings at or below this level
                    while heading_stack and heading_stack[-1][0] >= level:
                        heading_stack.pop()
                    heading_stack.append((level, title))

                    # If we already have accumulated content, flush before starting new section
                    if current_lines:
                        current_tokens = estimate_tokens("\n".join(current_lines))
                        if current_tokens >= self.target_size // 2:
                            flush_current_chunk()

            current_lines.append(line)

            # Check if current chunk has reached target size
            current_tokens = estimate_tokens("\n".join(current_lines))
            if current_tokens >= self.target_size and not in_code_block:
                flush_current_chunk()

        if current_lines:
            flush_current_chunk()

        return chunks
