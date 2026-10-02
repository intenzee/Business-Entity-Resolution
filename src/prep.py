"""Stage 0: normalise every source file once and cache it as parquet.

Output columns per record:
  id, country, raw_name, raw_addr,
  nm    - all folded name tokens (space-joined)
  core  - name tokens minus legal forms / stopwords
  legal - canonical legal forms present
  addr  - canonicalised address tokens
  nums  - numeric tokens from address
  lm    - landmark flag
  indic - 1 if the raw name used a non-Latin script (transliteration path)
"""
import os
import sys
from multiprocessing import Pool

import polars as pl

sys.path.insert(0, os.path.dirname(__file__))
from norm import name_tokens, addr_tokens  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # code/business_entity_resolution
DATA = os.environ.get("BER_DATA", os.path.join(_ROOT, "..", "..", "dataset"))
WORK = os.environ.get("BER_WORK", os.path.join(_ROOT, "work"))


def read_tsv(path):
    return pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False).fill_null("")


def _proc(args):
    names, addrs = args
    out = ([], [], [], [], [], [], [])
    for n, a in zip(names, addrs):
        _, core, legal = name_tokens(n)
        allt = name_tokens(n)[0]
        at, nums, lm = addr_tokens(a)
        out[0].append(" ".join(allt)); out[1].append(" ".join(core)); out[2].append(" ".join(legal))
        out[3].append(" ".join(at)); out[4].append(" ".join(nums)); out[5].append(lm)
        out[6].append(int(any(ord(ch) > 0x24F for ch in n)))
    return out


def prep(split, src, pool):
    df = read_tsv(f"{DATA}/{split}/{split}_source{src}.tsv")
    names, addrs = df["business_name"].to_list(), df["business_address"].to_list()
    step = 50_000
    chunks = [(names[i:i + step], addrs[i:i + step]) for i in range(0, len(names), step)]
    cols = [[] for _ in range(7)]
    for r in pool.imap(_proc, chunks):
        for c, v in zip(cols, r):
            c.extend(v)
    out = df.rename({"entity_id": "id", "business_name": "raw_name", "business_address": "raw_addr"}).with_columns(
        pl.Series("nm", cols[0]), pl.Series("core", cols[1]), pl.Series("legal", cols[2]),
        pl.Series("addr", cols[3]), pl.Series("nums", cols[4]), pl.Series("lm", cols[5], dtype=pl.Int8),
        pl.Series("indic", cols[6], dtype=pl.Int8))
    out.write_parquet(f"{WORK}/{split}_s{src}.parquet")
    print(split, src, out.height, flush=True)


if __name__ == "__main__":
    os.makedirs(WORK, exist_ok=True)
    with Pool(7) as pool:
        for split in ["train", "test"]:
            for src in ["1", "2", "3"]:
                prep(split, src, pool)
