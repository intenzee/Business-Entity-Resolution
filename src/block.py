"""Stage 1: multi-view candidate generation (blocking).

For every country label present in Source 1 (open set, never hard-coded) and for each
target source (S2, S3) separately, two TF-IDF views are indexed and searched with a
multithreaded sparse top-k matmul (sparse_dot_topn):

  * word view  - name core tokens (n_) + address tokens (a_), IDF weighted
  * char view  - character 3-grams of the space-free core name (catches typos,
                 "juarezswaim.com", "#shahdaralearning", token reordering)

The union of both views' top-k lists is the candidate set; the two cosine scores and
the per-view ranks are kept as model features.
"""
import os
import numpy as np
import polars as pl
import scipy.sparse as sp
from sklearn.feature_extraction.text import HashingVectorizer
from sparse_dot_topn import sp_matmul_topn

NF = 1 << 22


def _bigrams(toks, pre):
    return [pre + a + "_" + b for a, b in zip(toks, toks[1:])]


def _wb(doc):
    n, a = doc.split("\t")
    nt, at = n.split(), a.split()
    return ["n_" + x for x in nt] + ["a_" + x for x in at] + _bigrams(nt, "nb_") + _bigrams(at, "ab_") + \
        (["ns_" + "".join(sorted(nt))] if nt else [])


def _word_docs(df):
    return df.select(pl.concat_str(["core", "addr"], separator="\t"))[:, 0].to_list()


def _word_docs_old(df):
    return df.select(pl.concat_str([
        pl.col("core").str.replace_all(r"(\S+)", "n_$1"),
        pl.col("addr").str.replace_all(r"(\S+)", "a_$1")], separator=" "))[:, 0].to_list()


def _char_docs(df):
    return df["core"].str.replace_all(" ", "").to_list()


def _tfidf(docs_q, docs_t, vec, max_df):
    Q = vec.transform(docs_q).tocsr()
    T = vec.transform(docs_t).tocsr()
    Q.data[:] = 1.0
    T.data[:] = 1.0
    n = Q.shape[0] + T.shape[0]
    df = np.bincount(Q.indices, minlength=NF) + np.bincount(T.indices, minlength=NF)
    idf = np.log((1 + n) / (1 + df)) + 1.0
    idf[df > max(max_df * n, 50)] = 0.0  # stop-features: too common to discriminate, and expensive
    D = sp.diags(idf.astype(np.float32))
    Q = (Q.astype(np.float32) @ D).tocsr()
    T = (T.astype(np.float32) @ D).tocsr()
    for M in (Q, T):
        M.eliminate_zeros()
        nr = np.sqrt(np.asarray(M.multiply(M).sum(1)).ravel())
        nr[nr == 0] = 1
        M.data /= np.repeat(nr, np.diff(M.indptr)).astype(np.float32)
    return Q, T


def _topk(Q, T, k, thr):
    C = sp_matmul_topn(Q, T.T.tocsr(), top_n=k, threshold=thr, sort=True, n_threads=os.cpu_count())
    C = C.tocoo()
    return C.row.astype(np.int32), C.col.astype(np.int32), C.data.astype(np.float32)


WORD = HashingVectorizer(analyzer=_wb, n_features=NF, alternate_sign=False, norm=None, dtype=np.float32)
CHAR = HashingVectorizer(analyzer="char", ngram_range=(4, 4), n_features=NF, alternate_sign=False, norm=None,
                         dtype=np.float32)


def _pair_cos(Q, T, qi, ti):
    vals = np.zeros(len(qi), dtype=np.float32)
    B = 500_000
    for i in range(0, len(qi), B):
        vals[i:i + B] = np.asarray(Q[qi[i:i + B]].multiply(T[ti[i:i + B]]).sum(1)).ravel()
    return vals


MAXDF_W, MAXDF_C = 0.002, 0.002


def block(s1, tgt, k_word=12, k_char=8):
    """s1, tgt: prepped frames of one country. Returns pairs frame (qi, ti, w, c, rw, rc)."""
    Qw, Tw = _tfidf(_word_docs(s1), _word_docs(tgt), WORD, MAXDF_W)
    r, c, v = _topk(Qw, Tw, k_word, 0.05)
    pw = pl.DataFrame({"qi": r, "ti": c, "w": v})
    Qc, Tc = _tfidf(_char_docs(s1), _char_docs(tgt), CHAR, MAXDF_C)
    r, c, v = _topk(Qc, Tc, k_char, 0.2)
    pc = pl.DataFrame({"qi": r, "ti": c, "c": v})
    p = pw.join(pc, on=["qi", "ti"], how="full", coalesce=True)
    del pw, pc
    for name, Q, T in (("w", Qw, Tw), ("c", Qc, Tc)):
        m = p[name].is_null().to_numpy()
        vals = p[name].fill_null(0.0).to_numpy().copy()
        vals[m] = _pair_cos(Q, T, p["qi"].to_numpy()[m], p["ti"].to_numpy()[m])
        p = p.with_columns(pl.Series(name, vals))
    del Qw, Tw, Qc, Tc
    return p
