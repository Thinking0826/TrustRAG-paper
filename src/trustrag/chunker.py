"""Build citation-preserving retrieval chunks from parsed PDF text blocks."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "1.0"
DEFAULT_MAX_CHARACTERS = 1400
DEFAULT_OVERLAP_CHARACTERS = 180


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load non-empty JSONL records."""
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}") from error
    if not records:
        raise ValueError(f"No records found in {path}")
    return records


def choose_split_point(text: str, limit: int) -> int:
    """Prefer a sentence or line boundary near the character limit."""
    if len(text) <= limit:
        return len(text)

    minimum = max(limit // 2, 1)
    candidates = ("\n", "。", "！", "？", ". ", "! ", "? ", "; ", "；")
    best = -1
    for marker in candidates:
        position = text.rfind(marker, minimum, limit + 1)
        if position >= 0:
            best = max(best, position + len(marker))
    return best if best > 0 else limit


def split_block(record: dict[str, Any], max_characters: int) -> list[dict[str, Any]]:
    """Split an oversized block while retaining its citation metadata."""
    remaining = record["text"].strip()
    segments: list[dict[str, Any]] = []
    segment_index = 0

    while remaining:
        split_at = choose_split_point(remaining, max_characters)
        text = remaining[:split_at].strip()
        remaining = remaining[split_at:].strip()
        if not text:
            continue
        segments.append(
            {
                "text": text,
                "block_id": record["block_id"],
                "page_number": record["page_number"],
                "bbox": record["bbox"],
                "segment_index": segment_index,
            }
        )
        segment_index += 1

    return segments


def joined_length(units: list[dict[str, Any]]) -> int:
    """Return the length produced by joining units with blank lines."""
    if not units:
        return 0
    return sum(len(unit["text"]) for unit in units) + 2 * (len(units) - 1)


def trailing_overlap(
    units: list[dict[str, Any]], overlap_characters: int
) -> list[dict[str, Any]]:
    """Retain complete trailing units up to the requested overlap size."""
    selected: list[dict[str, Any]] = []
    for unit in reversed(units):
        candidate = [unit, *selected]
        if joined_length(candidate) > overlap_characters:
            break
        selected = candidate
    return selected


def make_chunk(
    document_id: str,
    source_file: str,
    index: int,
    units: list[dict[str, Any]],
    max_characters: int,
    overlap_characters: int,
) -> dict[str, Any]:
    """Create one retrieval chunk with deduplicated citation spans."""
    seen_spans: set[tuple[str, int]] = set()
    source_spans: list[dict[str, Any]] = []
    for unit in units:
        key = (unit["block_id"], unit["segment_index"])
        if key in seen_spans:
            continue
        seen_spans.add(key)
        source_spans.append(
            {
                "block_id": unit["block_id"],
                "page_number": unit["page_number"],
                "bbox": unit["bbox"],
                "segment_index": unit["segment_index"],
            }
        )

    pages = sorted({span["page_number"] for span in source_spans})
    text = "\n\n".join(unit["text"] for unit in units)
    return {
        "schema_version": SCHEMA_VERSION,
        "document_id": document_id,
        "chunk_id": f"{document_id}-c{index:05d}",
        "source_file": source_file,
        "page_start": pages[0],
        "page_end": pages[-1],
        "source_spans": source_spans,
        "text": text,
        "character_count": len(text),
        "chunking": {
            "method": "block_greedy_character",
            "max_characters": max_characters,
            "overlap_characters": overlap_characters,
        },
    }


def build_chunks(
    records: list[dict[str, Any]],
    max_characters: int = DEFAULT_MAX_CHARACTERS,
    overlap_characters: int = DEFAULT_OVERLAP_CHARACTERS,
) -> list[dict[str, Any]]:
    """Build ordered chunks from parsed block records."""
    if max_characters <= 0:
        raise ValueError("max_characters must be positive")
    if not 0 <= overlap_characters < max_characters:
        raise ValueError("overlap_characters must be between 0 and max_characters")

    document_ids = {record["document_id"] for record in records}
    source_files = {record["source_file"] for record in records}
    if len(document_ids) != 1 or len(source_files) != 1:
        raise ValueError("Input JSONL must contain exactly one document")

    document_id = next(iter(document_ids))
    source_file = next(iter(source_files))
    units = [
        segment
        for record in records
        for segment in split_block(record, max_characters=max_characters)
    ]

    chunks: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []

    for unit in units:
        if current and joined_length([*current, unit]) > max_characters:
            chunks.append(
                make_chunk(
                    document_id,
                    source_file,
                    len(chunks),
                    current,
                    max_characters,
                    overlap_characters,
                )
            )
            current = trailing_overlap(current, overlap_characters)
            while current and joined_length([*current, unit]) > max_characters:
                current.pop(0)
        current.append(unit)

    if current:
        chunks.append(
            make_chunk(
                document_id,
                source_file,
                len(chunks),
                current,
                max_characters,
                overlap_characters,
            )
        )

    return chunks


def write_jsonl_atomic(path: Path, records: list[dict[str, Any]]) -> None:
    """Write JSONL atomically."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as target:
        for record in records:
            target.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(path)


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON atomically."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as target:
        json.dump(payload, target, ensure_ascii=False, indent=2)
        target.write("\n")
    temporary.replace(path)


def build_parser() -> argparse.ArgumentParser:
    """Build command-line arguments."""
    default_data_dir = Path(os.environ.get("TRUSTRAG_DATA_DIR", "data"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("blocks", type=Path, help="Block JSONL produced by the PDF parser")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_data_dir / "processed",
        help="Directory for chunk JSONL and manifest JSON",
    )
    parser.add_argument("--max-characters", type=int, default=DEFAULT_MAX_CHARACTERS)
    parser.add_argument("--overlap-characters", type=int, default=DEFAULT_OVERLAP_CHARACTERS)
    return parser


def cli() -> None:
    """Run the chunk builder command."""
    arguments = build_parser().parse_args()
    input_path = arguments.blocks.expanduser().resolve()
    output_dir = arguments.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_jsonl(input_path)
    chunks = build_chunks(
        records,
        max_characters=arguments.max_characters,
        overlap_characters=arguments.overlap_characters,
    )

    suffix = ".blocks.jsonl"
    stem = input_path.name[: -len(suffix)] if input_path.name.endswith(suffix) else input_path.stem
    chunks_path = output_dir / f"{stem}.chunks.jsonl"
    manifest_path = output_dir / f"{stem}.chunks.manifest.json"
    write_jsonl_atomic(chunks_path, chunks)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "document_id": records[0]["document_id"],
        "source_file": records[0]["source_file"],
        "input_block_count": len(records),
        "chunk_count": len(chunks),
        "max_chunk_characters_observed": max(chunk["character_count"] for chunk in chunks),
        "mean_chunk_characters": round(
            sum(chunk["character_count"] for chunk in chunks) / len(chunks), 2
        ),
        "chunking": {
            "method": "block_greedy_character",
            "max_characters": arguments.max_characters,
            "overlap_characters": arguments.overlap_characters,
        },
    }
    write_json_atomic(manifest_path, manifest)

    print(f"Input blocks: {manifest['input_block_count']}")
    print(f"Chunks: {manifest['chunk_count']}")
    print(f"Mean characters: {manifest['mean_chunk_characters']}")
    print(f"Max characters: {manifest['max_chunk_characters_observed']}")
    print(f"Chunks file: {chunks_path}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    cli()
