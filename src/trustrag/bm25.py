"""Dependency-free BM25 retrieval for TrustRAG chunks."""

from __future__ import annotations

import argparse
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from trustrag.chunker import load_jsonl
from trustrag.dense_search import rank_scores

TOKEN_PATTERN = re.compile(r"[a-z0-9]+|[\u3400-\u4dbf\u4e00-\u9fff]+")


def tokenize(text: str) -> list[str]:
    """Tokenize Latin words and add Chinese character unigrams and bigrams."""
    tokens: list[str] = []
    for match in TOKEN_PATTERN.finditer(text.lower()):
        token = match.group(0)
        if "\u3400" <= token[0] <= "\u9fff":
            characters = list(token)
            tokens.extend(characters)
            tokens.extend(
                characters[index] + characters[index + 1]
                for index in range(len(characters) - 1)
            )
        else:
            tokens.append(token)
    return tokens


@dataclass(slots=True)
class BM25Index:
    """A compact in-memory Okapi BM25 index."""

    term_frequencies: list[Counter[str]]
    document_lengths: np.ndarray
    average_document_length: float
    inverse_document_frequency: dict[str, float]
    k1: float = 1.5
    b: float = 0.75

    @classmethod
    def from_texts(
        cls,
        texts: list[str],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> BM25Index:
        """Build a BM25 index from document texts."""
        if not texts:
            raise ValueError("texts is empty")
        if k1 <= 0:
            raise ValueError("k1 must be positive")
        if not 0 <= b <= 1:
            raise ValueError("b must be between 0 and 1")

        term_frequencies = [Counter(tokenize(text)) for text in texts]
        document_lengths = np.asarray(
            [sum(frequencies.values()) for frequencies in term_frequencies],
            dtype=np.float32,
        )
        average_document_length = float(document_lengths.mean())
        document_frequency: Counter[str] = Counter()
        for frequencies in term_frequencies:
            document_frequency.update(frequencies.keys())
        document_count = len(texts)
        inverse_document_frequency = {
            term: math.log(1 + (document_count - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }
        return cls(
            term_frequencies=term_frequencies,
            document_lengths=document_lengths,
            average_document_length=average_document_length,
            inverse_document_frequency=inverse_document_frequency,
            k1=k1,
            b=b,
        )

    def scores(self, query: str) -> np.ndarray:
        """Score every indexed document for one query."""
        scores = np.zeros(len(self.term_frequencies), dtype=np.float32)
        query_terms = set(tokenize(query))
        if not query_terms or self.average_document_length == 0:
            return scores

        for term in query_terms:
            inverse_document_frequency = self.inverse_document_frequency.get(term)
            if inverse_document_frequency is None:
                continue
            for index, frequencies in enumerate(self.term_frequencies):
                term_frequency = frequencies.get(term, 0)
                if term_frequency == 0:
                    continue
                length_ratio = self.document_lengths[index] / self.average_document_length
                denominator = term_frequency + self.k1 * (1 - self.b + self.b * length_ratio)
                scores[index] += inverse_document_frequency * (
                    term_frequency * (self.k1 + 1) / denominator
                )
        return scores


def positive_ranking(scores: np.ndarray, limit: int) -> list[int]:
    """Return positive-score document indices in stable descending order."""
    if limit <= 0:
        raise ValueError("limit must be positive")
    return [
        int(index)
        for index in rank_scores(scores, len(scores))
        if scores[index] > 0
    ][:limit]


def build_parser() -> argparse.ArgumentParser:
    """Build command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("chunks", type=Path, help="Chunk JSONL file")
    parser.add_argument("query", help="Natural-language search query")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--preview-characters", type=int, default=500)
    return parser


def cli() -> None:
    """Run BM25 search and print citation-aware results."""
    arguments = build_parser().parse_args()
    chunks = load_jsonl(arguments.chunks.expanduser().resolve())
    index = BM25Index.from_texts([chunk["text"] for chunk in chunks])
    scores = index.scores(arguments.query)
    ranking = positive_ranking(scores, arguments.top_k)
    if not ranking:
        print("No lexical matches found")
        return
    for rank, row_index in enumerate(ranking, start=1):
        chunk = chunks[row_index]
        pages = (
            str(chunk["page_start"])
            if chunk["page_start"] == chunk["page_end"]
            else f"{chunk['page_start']}-{chunk['page_end']}"
        )
        preview = chunk["text"][: arguments.preview_characters].replace("\n", " ")
        print(
            f"#{rank} bm25={scores[row_index]:.4f} source={chunk['source_file']} "
            f"pages={pages} chunk={chunk['chunk_id']}"
        )
        print(preview)
        print("-" * 80)


if __name__ == "__main__":
    cli()
