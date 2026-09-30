"""
Byte-level Byte-Pair Encoding (the GPT-2 family's tokenizer), from scratch.

Training: start from the 256 byte values, then repeatedly merge the most frequent
adjacent pair into a new token. Text is first split into "words" with a regex so merges
never cross word/punctuation boundaries. Counting happens over *unique* words weighted
by frequency, and pair counts are updated incrementally, so training is fast.

Also includes a trivial CharTokenizer with the same interface.
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict

SPLIT_PATTERN = re.compile(r"""'(?:s|t|re|ve|m|ll|d)| ?[A-Za-z]+| ?[0-9]{1,3}| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+""")


class CharTokenizer:
    def __init__(self, chars):
        self.chars = sorted(set(chars))
        self.stoi = {c: i for i, c in enumerate(self.chars)}

    @classmethod
    def from_text(cls, text: str):
        return cls(text)

    @property
    def vocab_size(self):
        return len(self.chars)

    def encode(self, text: str) -> list[int]:
        return [self.stoi[c] for c in text if c in self.stoi]

    def decode(self, ids) -> str:
        return "".join(self.chars[i] for i in ids)

    def token_strings(self) -> list[str]:
        return list(self.chars)

    def to_dict(self):
        return {"type": "char", "chars": "".join(self.chars)}


class BPETokenizer:
    def __init__(self, merges: list[tuple[int, int]] | None = None):
        self.merges: list[tuple[int, int]] = [tuple(m) for m in (merges or [])]
        self._build()

    def _build(self):
        self.ranks = {pair: 256 + i for i, pair in enumerate(self.merges)}
        self.vocab = {i: bytes([i]) for i in range(256)}
        for pair, idx in self.ranks.items():
            self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]
        self._cache: dict[str, list[int]] = {}

    @property
    def vocab_size(self):
        return 256 + len(self.merges)

    # ---------------------------------------------------------------- training
    @classmethod
    def train(cls, text: str, vocab_size: int, verbose: bool = False) -> "BPETokenizer":
        assert vocab_size >= 256
        word_freq = Counter(SPLIT_PATTERN.findall(text))
        words = [list(w.encode("utf-8")) for w in word_freq]
        freqs = list(word_freq.values())

        pair_counts: Counter = Counter()
        where: dict[tuple, set] = defaultdict(set)  # pair -> indices of words containing it
        for wi, w in enumerate(words):
            for pair in zip(w, w[1:]):
                pair_counts[pair] += freqs[wi]
                where[pair].add(wi)

        merges = []
        for new_id in range(256, vocab_size):
            if not pair_counts:
                break
            best = max(pair_counts, key=lambda p: (pair_counts[p], -p[0], -p[1]))
            if pair_counts[best] < 2:
                break
            merges.append(best)
            for wi in list(where[best]):
                w, f = words[wi], freqs[wi]
                for pair in zip(w, w[1:]):  # remove this word's old pairs
                    pair_counts[pair] -= f
                    if pair_counts[pair] <= 0:
                        del pair_counts[pair]
                merged, i = [], 0
                while i < len(w):
                    if i + 1 < len(w) and (w[i], w[i + 1]) == best:
                        merged.append(new_id)
                        i += 2
                    else:
                        merged.append(w[i])
                        i += 1
                words[wi] = merged
                for pair in zip(merged, merged[1:]):  # add the new ones
                    pair_counts[pair] += f
                    where[pair].add(wi)
            del where[best]
            if verbose and (new_id - 255) % 100 == 0:
                print(f"merge {new_id - 255}/{vocab_size - 256}")
        return cls(merges)

    # ---------------------------------------------------------------- encode / decode
    def _encode_word(self, word: str) -> list[int]:
        cached = self._cache.get(word)
        if cached is not None:
            return cached
        ids = list(word.encode("utf-8"))
        while len(ids) >= 2:
            # merge the pair that was learned earliest (lowest rank) first
            pair = min(zip(ids, ids[1:]), key=lambda p: self.ranks.get(p, float("inf")))
            if pair not in self.ranks:
                break
            new, merged, i = self.ranks[pair], [], 0
            while i < len(ids):
                if i + 1 < len(ids) and (ids[i], ids[i + 1]) == pair:
                    merged.append(new)
                    i += 2
                else:
                    merged.append(ids[i])
                    i += 1
            ids = merged
        if len(self._cache) < 100_000:
            self._cache[word] = ids
        return ids

    def encode(self, text: str) -> list[int]:
        out = []
        for word in SPLIT_PATTERN.findall(text):
            out.extend(self._encode_word(word))
        return out

    def decode(self, ids) -> str:
        return b"".join(self.vocab[int(i)] for i in ids).decode("utf-8", errors="replace")

    def token_strings(self) -> list[str]:
        return [self.vocab[i].decode("utf-8", errors="replace") for i in range(self.vocab_size)]

    # ---------------------------------------------------------------- persistence
    def to_dict(self):
        return {"type": "bpe", "merges": [list(m) for m in self.merges]}

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def load(cls, path) -> "BPETokenizer":
        with open(path) as f:
            return cls(json.load(f)["merges"])


def tokenizer_from_dict(d: dict):
    return CharTokenizer(d["chars"]) if d["type"] == "char" else BPETokenizer(d["merges"])
