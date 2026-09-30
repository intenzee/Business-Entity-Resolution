"""End-to-end pipeline driver.

  python src/pipeline.py candidates train   # blocking + features for all train S1
  python src/pipeline.py candidates test    # blocking + features for all test S1
  python src/pipeline.py train              # LightGBM on train candidates, OOF eval
  python src/pipeline.py predict            # score test, decode, write output/*.tsv

Work is done per (country, target source) so memory stays bounded on an 8 GB laptop.
Country is iterated over whatever labels Source 1 contains (open set).
"""
import gc
import json
import os
import sys
import time

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(__file__))
from block import block  # noqa: E402
from features import pair_features, context_features  # noqa: E402

WORK = os.environ.get("BER_WORK", "/Users/tanmayroy/Downloads/ber/work")
OUT = os.environ.get("BER_OUT", "/Users/tanmayroy/Downloads/ber/output")
K_WORD, K_CHAR = int(os.environ.get("BER_KW", 10)), int(os.environ.get("BER_KC", 6))
TR_PCT = int(os.environ.get("BER_TRPCT", 12))  # stage-1 fold width (% of S1 hash space)
VPCT = int(os.environ.get("BER_VPCT", 6))       # validation = hash folds [50-VPCT, 50)
TR2 = int(os.environ.get("BER_TR2", 16))        # stage-2 training folds [0, TR2)
VLO = 50 - VPCT
NM_PCT = int(os.environ.get("BER_NMPCT", 100))  # % of never-matched (decoy) records kept
COLS = ["id", "country", "core", "nm", "legal", "addr", "nums", "lm", "indic"]


def apply_map(df, maps):
    """Rewrite tokens with the data-mined dictionary, then rebuild the core name."""
    from norm import LEGAL, LEGAL_FORM, LEGAL_CANON
    if maps.get("translit"):
        df = df.with_columns(nm=pl.when(pl.col("indic") == 1).then(
            pl.col("nm").str.split(" ").list.eval(pl.element().replace(maps["translit"])).list.join(" "))
            .otherwise(pl.col("nm")))
    out = df.with_columns(
        pl.col("nm").str.split(" ").list.eval(pl.element().replace(maps["nm"])).list.join(" "),
        pl.col("addr").str.split(" ").list.eval(pl.element().replace(maps["addr"])).list.join(" "),
    )
    # self-trained, country-specific rules (see selftrain.py); applied only to that label
    for c, rules in maps.get("by_country", {}).items():
        for fld in ("nm", "addr"):
            if rules.get(fld):
                out = out.with_columns(pl.when(pl.col("country") == c).then(
                    pl.col(fld).str.split(" ").list.eval(pl.element().replace(rules[fld])).list.join(" "))
                    .otherwise(pl.col(fld)).alias(fld))
    legal = list(LEGAL)
    out = out.with_columns(
        core=pl.col("nm").str.split(" ").list.eval(pl.element().filter(~pl.element().is_in(legal))).list.join(" "),
        legal=pl.col("nm").str.split(" ").list.eval(
            pl.element().filter(pl.element().is_in(list(LEGAL_FORM))).replace(LEGAL_CANON)).list.unique().list.sort().list.join(" "),
    )
    return out.with_columns(core=pl.when(pl.col("core") == "").then(pl.col("nm")).otherwise(pl.col("core")))


def load(split, src, maps, country=None):
    df = pl.read_parquet(f"{WORK}/{split}_s{src}.parquet", columns=COLS)
    if country is not None:
        df = df.filter(pl.col("country") == country)
    return apply_map(df, maps) if maps else df


def _hash100(col):
    return (pl.col(col).hash(seed=7) % 100).cast(pl.Int16)


def train_subsample(s1, tg, gt):
    """Mimic the test distribution on train. Test has ~5.76 S2/S3 records per S1 entity vs
    4.67 in train, i.e. extra 'orphan' records whose S1 anchor is absent. We keep S1 set A
    (50%) as queries, keep the matched records of set B (16%) as orphans (their S1 is
    dropped), and 50% of never-matched records."""
    a = s1.filter(_hash100("id") < 50)
    b_ids = s1.filter((_hash100("id") >= 50) & (_hash100("id") < 66))["id"]
    ab_t = gt.filter(pl.col("q").is_in(pl.concat([a["id"], b_ids]).implode()))["t"]
    all_matched = gt["t"]
    keep = tg.filter(pl.col("id").is_in(ab_t.implode())
                     | (~pl.col("id").is_in(all_matched.implode()) & (_hash100("id") < NM_PCT)))
    return a, keep


def candidates(split):
    maps = json.load(open(f"{WORK}/token_map.json"))
    s1 = load(split, "1", maps)
    gt = None
    if split == "train":
        from mine import load_gt_pairs
        gt = load_gt_pairs()
        s1, _ = train_subsample(s1, s1.head(0), gt)
    countries = s1["country"].unique().sort().to_list()
    for src in ["2", "3"]:
        tg = load(split, src, maps)
        if split == "train":
            _, tg = train_subsample(load(split, "1", maps).select(COLS), tg, gt)
        for c in countries:
            fn = f"{WORK}/cand_{split}_{c}_s{src}.parquet"
            if os.path.exists(fn):
                continue
            t0 = time.time()
            q = s1.filter(pl.col("country") == c)
            t = tg.filter(pl.col("country") == c)
            p = block(q, t, K_WORD, K_CHAR)
            p = p.with_columns(pl.lit(int(src), dtype=pl.Int8).alias("src"))
            qq = q.select([pl.col(x).alias("q_" + x) for x in COLS]).with_row_index("qi")
            tt = t.select([pl.col(x).alias("t_" + x) for x in COLS]).with_row_index("ti")
            p = p.with_columns(pl.col("qi").cast(pl.UInt32), pl.col("ti").cast(pl.UInt32))
            parts = []
            B = 2_000_000
            for i in range(0, p.height, B):
                pp = p.slice(i, B).join(qq, on="qi").join(tt, on="ti")
                f = pair_features(pp).with_columns(q=pp["q_id"], t=pp["t_id"])
                parts.append(f)
                del pp
                gc.collect()
            f = pl.concat(parts)
            f = context_features(f)
            f.write_parquet(fn)
            print(split, c, src, "pairs", f.height, "per q", round(f.height / q.height, 1),
                  f"{time.time() - t0:.0f}s", flush=True)
            del p, f, parts, q, t
            gc.collect()
        del tg
        gc.collect()


DROP = {"qi", "ti", "q", "t", "y", "sc"}


def _addr_keys(df):
    """Address keys: exact (sorted address tokens) and street-level (numbers removed)."""
    return df.with_columns(
        ak=pl.when(pl.col("addr") == "").then(None).otherwise(pl.col("addr").str.split(" ").list.sort().list.join(" ")),
        sk=pl.when(pl.col("addr") == "").then(None).otherwise(
            pl.col("addr").str.split(" ").list.eval(pl.element().filter(~pl.element().str.contains(r"^\d+$")))
            .list.sort().list.join(" ")))


def name_freq_tables(split, maps=None):
    """Name- and address-uniqueness statistics within a country.

    Names: how many S1 entities / target records share a core name. A unique name with an
    empty address is a safe match; a name shared by 20 businesses is not.
    Addresses: how many S1 entities / target records share the exact address, how many
    distinct names live there, and how dense the street is. Where several businesses share
    one address (common in some countries), a same-address match needs name agreement."""
    fn = f"{WORK}/nfa_{split}.parquet"
    if os.path.exists(fn):
        return pl.read_parquet(fn)
    maps = maps or json.load(open(f"{WORK}/token_map.json"))
    cols = ["id", "country", "core", "addr"]
    s1 = load(split, "1", maps).select(cols)
    tg = pl.concat([load(split, s, maps).select(cols) for s in "23"])
    if split == "train":
        from mine import load_gt_pairs
        gtp = load_gt_pairs()
        full1 = load(split, "1", maps).select(COLS)
        s1, _ = train_subsample(s1, s1.head(0), gtp)
        _, tg = train_subsample(full1, tg, gtp)
    s1, tg = _addr_keys(s1), _addr_keys(tg)
    s1c = s1.group_by("country", "core").len().rename({"len": "nf_s1"})
    tgc = tg.group_by("country", "core").len().rename({"len": "nf_t"})
    s1a = s1.drop_nulls("ak").group_by("country", "ak").agg(af_s1=pl.len(), afc_s1=pl.col("core").n_unique())
    tga = tg.drop_nulls("ak").group_by("country", "ak").agg(af_t=pl.len(), afc_t=pl.col("core").n_unique())
    s1s = s1.drop_nulls("sk").group_by("country", "sk").agg(sf_s1=pl.len())
    tgs = tg.drop_nulls("sk").group_by("country", "sk").agg(sf_t=pl.len())

    def stats(d):
        return (d.join(s1c, on=["country", "core"], how="left").join(tgc, on=["country", "core"], how="left")
                .join(s1a, on=["country", "ak"], how="left").join(tga, on=["country", "ak"], how="left")
                .join(s1s, on=["country", "sk"], how="left").join(tgs, on=["country", "sk"], how="left")
                .fill_null(0))
    stat_cols = ["nf_s1", "nf_t", "af_s1", "afc_s1", "af_t", "afc_t", "sf_s1", "sf_t"]
    q, t = stats(s1), stats(tg)
    out = pl.concat([
        q.select("id", *[pl.col(c).alias("q_" + c) for c in stat_cols], q_ak=pl.when(pl.col("ak").is_null()).then(pl.lit(0, pl.UInt64)).otherwise(pl.col("ak").hash())),
        t.select("id", *[pl.col(c).alias("t_" + c) for c in stat_cols], t_ak=pl.when(pl.col("ak").is_null()).then(pl.lit(1, pl.UInt64)).otherwise(pl.col("ak").hash()))],
        how="diagonal")
    out.write_parquet(fn)
    return out


NF_Q = ["q_nf_s1", "q_nf_t", "q_af_s1", "q_afc_s1", "q_af_t", "q_afc_t", "q_sf_s1", "q_sf_t"]
NF_T = ["t_nf_s1", "t_nf_t", "t_af_s1", "t_afc_s1", "t_af_t", "t_afc_t", "t_sf_s1", "t_sf_t"]


def enrich(f, nf):
    q = nf.filter(pl.col("q_nf_s1").is_not_null()).select(pl.col("id").alias("q"), *NF_Q, "q_ak")
    t = nf.filter(pl.col("t_nf_t").is_not_null()).select(pl.col("id").alias("t"), *NF_T, "t_ak")
    f = f.join(q, on="q", how="left").join(t, on="t", how="left")
    return f.with_columns(same_ak=(pl.col("q_ak") == pl.col("t_ak")).cast(pl.Int8)).drop("q_ak", "t_ak").fill_null(0)


def _files(split):
    return sorted(f for f in os.listdir(WORK) if f.startswith(f"cand_{split}_"))


def _labels():
    from mine import load_gt_pairs
    return load_gt_pairs().with_columns(y=pl.lit(1, dtype=pl.Int8))


def _fold(col="q"):
    # deterministic entity-level split: hash of the S1 id -> [0, 100)
    return (pl.col(col).hash(seed=7) % 100).cast(pl.Int16)


def train(train_pct=12, valid_pct=6):
    import lightgbm as lgb
    gt = _labels()
    nf = name_freq_tables("train")
    tr, va = [], []
    for fn in _files("train"):
        f = enrich(pl.read_parquet(f"{WORK}/{fn}"), nf).with_columns(_fold().alias("fold"), country=pl.lit(fn.split("_")[2]))
        tr.append(f.filter(pl.col("fold") < train_pct))
        va.append(f.filter((pl.col("fold") >= 50 - valid_pct)))
        del f
    tr = pl.concat(tr).join(gt, on=["q", "t"], how="left").with_columns(pl.col("y").fill_null(0))
    va = pl.concat(va).join(gt, on=["q", "t"], how="left").with_columns(pl.col("y").fill_null(0))
    feats = [c for c in tr.columns if c not in DROP | {"fold", "country"}]
    print("train rows", tr.height, "pos", tr["y"].mean(), "valid rows", va.height, "feats", len(feats), flush=True)
    params = dict(objective="binary", learning_rate=0.08, num_leaves=127, min_data_in_leaf=200,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  num_threads=os.cpu_count(), verbose=-1, max_bin=127)
    dtr = lgb.Dataset(tr.select(feats).to_numpy().astype(np.float32), tr["y"].to_numpy(), free_raw_data=True)
    dva = lgb.Dataset(va.select(feats).to_numpy().astype(np.float32), va["y"].to_numpy(), reference=dtr)
    m = lgb.train(params, dtr, 1500, valid_sets=[dva], callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)])
    m.save_model(f"{WORK}/lgb.txt")
    json.dump(feats, open(f"{WORK}/feats.json", "w"))
    va = va.with_columns(p=pl.Series(m.predict(va.select(feats).to_numpy().astype(np.float32), num_threads=os.cpu_count())))
    va.select("q", "t", "p", "y", "country", "src").write_parquet(f"{WORK}/valid_pred.parquet")
    imp = sorted(zip(m.feature_importance("gain"), feats), reverse=True)
    print("top features:", [(f, round(g / 1e3)) for g, f in imp[:25]])
    evaluate(valid_pct)


def evaluate(valid_pct=None):
    valid_pct = valid_pct or VPCT
    """Validation S1 = hash fold in [44, 50) of the query set A (never used for training)."""
    from decode import decode, macro_f05
    va = pl.read_parquet(f"{WORK}/valid_pred.parquet")
    gt = _labels()
    s1 = pl.read_parquet(f"{WORK}/train_s1.parquet", columns=["id", "country"]).rename({"id": "q"})
    va = va.drop("country", strict=False)
    s1v = s1.filter((_fold() >= 50 - valid_pct) & (_fold() < 50))
    truth = gt.join(s1v, on="q").select("q", "t")
    allq = s1v["q"].to_list()
    cand = va.select("q", "t").unique()
    ceiling = truth.join(cand, on=["q", "t"]).height / truth.height
    print(f"valid S1={len(allq)}  pair recall ceiling={ceiling:.4f}")
    for thr in [0.3, 0.5, 0.6, 0.7]:
        pr = va.filter(pl.col("p") > thr).select("q", "t")
        print(f"  threshold {thr}: F0.5={macro_f05(pr, truth, allq):.5f}")
    pr = va.filter(pl.col("p") == pl.col("p").max().over("t")).filter(pl.col("p") > 0.5).select("q", "t")
    print(f"  exclusive + 0.5: F0.5={macro_f05(pr, truth, allq):.5f}")
    for ex in [False, True]:
        pr = decode(va.select("q", "t", "p"), exclusive=ex)
        print(f"  expected-F decoder (exclusive={ex}): F0.5={macro_f05(pr, truth, allq):.5f}")
    best = {"mode": "thr", "thr": 0.5, "f": -1}
    for thr in [0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8]:
        pr = va.filter(pl.col("p") == pl.col("p").max().over("t")).filter(pl.col("p") > thr).select("q", "t")
        f = macro_f05(pr, truth, allq)
        print(f"  exclusive thr {thr}: {f:.5f}")
        if f > best["f"]:
            best = {"mode": "thr", "thr": thr, "f": f}
    for g in [1.0, 1.5, 2.0]:
        pr = decode(va.with_columns(p=pl.col("p") ** g).select("q", "t", "p"), exclusive=True)
        f = macro_f05(pr, truth, allq)
        print(f"  decoder pow {g}: {f:.5f}")
        if f > best["f"]:
            best = {"mode": "dec", "pow": g, "f": f}
    print("BEST", best)
    json.dump(best, open(f"{WORK}/decode_cfg.json", "w"))
    pr = decode(va.select("q", "t", "p"), exclusive=True)
    for c in s1v["country"].unique().to_list():
        qs = s1v.filter(pl.col("country") == c)["q"]
        print(f"    {c}: F0.5={macro_f05(pr.filter(pl.col('q').is_in(qs.implode())), truth.filter(pl.col('q').is_in(qs.implode())), qs.to_list()):.5f}")


LGB_PARAMS = dict(objective="binary", learning_rate=0.08, num_leaves=127, min_data_in_leaf=200,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  num_threads=os.cpu_count(), verbose=-1, max_bin=127)


def _fit(tr, va, feats, rounds=1500):
    import lightgbm as lgb
    dtr = lgb.Dataset(tr.select(feats).to_numpy().astype(np.float32), tr["y"].to_numpy())
    dva = lgb.Dataset(va.select(feats).to_numpy().astype(np.float32), va["y"].to_numpy(), reference=dtr)
    return lgb.train(LGB_PARAMS, dtr, rounds, valid_sets=[dva],
                     callbacks=[lgb.early_stopping(50), lgb.log_evaluation(200)])


def _pred(m, f, feats):
    return m.predict(f.select(feats).to_numpy().astype(np.float32), num_threads=os.cpu_count())


def prob_context(d):
    """Stage-2 list-wise features in probability space. d: q, t, src, p1."""
    return d.with_columns(
        p_rank_q=pl.col("p1").rank("ordinal", descending=True).over("q", "src").cast(pl.Int16),
        p_max_q=pl.col("p1").max().over("q", "src"),
        p_gap_q=pl.col("p1") - pl.col("p1").max().over("q", "src"),
        p_sum_q=pl.col("p1").sum().over("q", "src"),
        p_n50_q=(pl.col("p1") > 0.5).sum().over("q", "src").cast(pl.Int16),
        p_sum_q_all=pl.col("p1").sum().over("q"),
        p_n50_q_all=(pl.col("p1") > 0.5).sum().over("q").cast(pl.Int16),
        p_max_t=pl.col("p1").max().over("t"),
        p_gap_t=pl.col("p1") - pl.col("p1").max().over("t"),
        p_second_t=pl.col("p1").top_k(2).min().over("t"),
        p_sum_t=pl.col("p1").sum().over("t"),
    ).with_columns(
        # best probability this S1 entity reaches in the *other* source
        p_max_other=(pl.col("p_sum_q_all") * 0 + pl.col("p1").max().over("q")) ,
    )


def _load_train(nf, gt, cond):
    out = []
    for fn in _files("train"):
        f = enrich(pl.read_parquet(f"{WORK}/{fn}"), nf).with_columns(_fold().alias("fold"))
        f = f.filter(cond)
        out.append(f.join(gt, on=["q", "t"], how="left").with_columns(pl.col("y").fill_null(0)))
        del f
        gc.collect()
    return pl.concat(out)


def train2(tr_pct=None):
    tr_pct = tr_pct or TR_PCT
    """Stage 1 (two cross-fitted models) -> OOF p1 on every train candidate -> stage 2.
    Streams per file so peak memory stays within an 8 GB laptop."""
    gt = _labels()
    nf = name_freq_tables("train")
    feats = json.load(open(f"{WORK}/feats.json"))
    feats1 = feats + NF_Q + NF_T + ["same_ak"]
    json.dump(feats1, open(f"{WORK}/feats1.json", "w"))
    va_small = _load_train(nf, gt, pl.col("fold") >= 48)
    models = []
    for lo, hi, name in [(0, tr_pct, "lgb1.txt"), (tr_pct, 2 * tr_pct, "lgb1b.txt")]:
        if os.path.exists(f"{WORK}/{name}"):
            import lightgbm as lgb
            models.append(lgb.Booster(model_file=f"{WORK}/{name}"))
            continue
        tr = _load_train(nf, gt, (pl.col("fold") >= lo) & (pl.col("fold") < hi))
        print("stage1 fit", name, tr.height, flush=True)
        m = _fit(tr, va_small, feats1)
        m.save_model(f"{WORK}/{name}")
        models.append(m)
        del tr
        gc.collect()
    mA, mB = models
    if os.path.exists(f"{WORK}/ctx_train.parquet"):  # resume: OOF context already built
        ctx = pl.read_parquet(f"{WORK}/ctx_train.parquet")
    else:
        ctx = None
    small = [] if ctx is None else None
    for fn in (_files("train") if ctx is None else []):
        f = enrich(pl.read_parquet(f"{WORK}/{fn}"), nf).with_columns(_fold().alias("fold"))
        pa = _pred(mA, f, feats1).astype(np.float32)
        isA = (f["fold"] < tr_pct).to_numpy()
        if isA.any():
            pa[isA] = _pred(mB, f.filter(pl.Series(isA)), feats1)
        small.append(f.select("q", "t", "src").with_columns(p1=pl.Series(pa)))
        del f
        gc.collect()
    if ctx is None:
        ctx = prob_context(pl.concat(small))
        del small
        ctx.write_parquet(f"{WORK}/ctx_train.parquet")
    ctx = ctx.with_columns(pl.col(pl.Float64).cast(pl.Float32))
    feats2 = feats1 + ["p1"] + [c for c in ctx.columns if c.startswith("p_")]
    json.dump(feats2, open(f"{WORK}/feats2.json", "w"))

    def with_ctx(cond):
        return _load_train(nf, gt, cond).join(ctx, on=["q", "t", "src"], how="left")

    tr = with_ctx(pl.col("fold") < TR2)
    va = with_ctx(pl.col("fold") >= VLO)
    print("stage2 train", tr.height, "valid", va.height, flush=True)
    m2 = _fit(tr, va, feats2)
    del tr
    gc.collect()
    m2.save_model(f"{WORK}/lgb2.txt")
    va = va.with_columns(p=pl.Series(_pred(m2, va, feats2)))
    va.select("q", "t", "p", "p1", "y", "src").write_parquet(f"{WORK}/valid_pred.parquet")
    imp = sorted(zip(m2.feature_importance("gain"), feats2), reverse=True)
    print("stage2 top features:", [(f, round(g / 1e3)) for g, f in imp[:20]], flush=True)
    evaluate()


def _target_recs(split, ids, maps, country=None):
    return pl.concat([load(split, src, maps, country).select("id", "core", "addr", "nums").filter(pl.col("id").is_in(ids.implode()))
                      for src in "23"])


def train3(extra=None):
    """Stage 2 + collective S2<->S3 anchor features (see anchor.py).
    extra: optional frame of pseudo-labelled test rows (France self-training)."""
    from anchor import anchor_features, ANCHOR_FEATS
    gt = _labels()
    nf = name_freq_tables("train")
    maps = json.load(open(f"{WORK}/token_map.json"))
    TR_HI = int(os.environ.get("BER_TR3", 13))
    ctx = pl.scan_parquet(f"{WORK}/ctx_train.parquet").filter((_fold() < TR_HI) | (_fold() >= VLO)).collect()
    ctx = ctx.with_columns(pl.col(pl.Float64).cast(pl.Float32))
    afn = f"{WORK}/anc_train.parquet"
    if not os.path.exists(afn):
        keep = ctx.filter((_fold() < TR_HI) | (_fold() >= VLO))
        recs = _target_recs("train", keep["t"].unique(), maps)
        anchor_features(keep.select("q", "t", "src", "p1"), recs).write_parquet(afn)
        del keep, recs
        gc.collect()
    anc = pl.scan_parquet(afn).filter((_fold() < TR_HI) | (_fold() >= VLO)).collect()
    feats2 = json.load(open(f"{WORK}/feats2.json"))
    feats3 = feats2 + ANCHOR_FEATS
    json.dump(feats3, open(f"{WORK}/feats3.json", "w"))

    def rows(cond):
        return (_load_train(nf, gt, cond).join(ctx, on=["q", "t", "src"], how="left")
                .join(anc, on=["q", "t"], how="left"))

    tr = rows(pl.col("fold") < TR_HI)
    if extra is not None:
        tr = pl.concat([tr.select(feats3 + ["y"]), extra.select(feats3 + ["y"])])
    va = rows(pl.col("fold") >= VLO)
    print("stage3 train", tr.height, "valid", va.height, flush=True)
    m3 = _fit(tr, va, feats3)
    del tr
    gc.collect()
    m3.save_model(f"{WORK}/lgb3.txt")
    va = va.with_columns(p=pl.Series(_pred(m3, va, feats3)))
    va.select("q", "t", "p", "p1", "y", "src").write_parquet(f"{WORK}/valid_pred.parquet")
    imp = sorted(zip(m3.feature_importance("gain"), feats3), reverse=True)
    print("stage3 top features:", [(f, round(g / 1e3)) for g, f in imp[:20]], flush=True)
    evaluate()


def predict2(thr=None, stage="2", only=None, tag="score2"):
    import lightgbm as lgb
    from decode import decode
    m1 = lgb.Booster(model_file=f"{WORK}/lgb1.txt")
    m2 = lgb.Booster(model_file=f"{WORK}/lgb{stage}.txt")
    feats1 = json.load(open(f"{WORK}/feats1.json"))
    feats2 = json.load(open(f"{WORK}/feats{stage}.json"))
    use_anchor = any(f.startswith("anc") for f in feats2)
    maps = json.load(open(f"{WORK}/token_map.json"))
    nf = name_freq_tables("test")
    # Candidates never cross countries, so every context feature is country-local:
    # process one country at a time and spill scores to disk to bound memory.
    countries = sorted({fn.split("_")[2] for fn in _files("test")})
    for c in countries:
        out_fn = f"{WORK}/{tag}_{c}.parquet"
        if os.path.exists(out_fn) or (only and c not in only):
            continue
        fns = [fn for fn in _files("test") if fn.split("_")[2] == c]
        small = []
        for fn in fns:
            f = enrich(pl.read_parquet(f"{WORK}/{fn}"), nf)
            small.append(f.select("q", "t", "src").with_columns(p1=pl.Series(_pred(m1, f, feats1), dtype=pl.Float32)))
            del f
            gc.collect()
        small = pl.concat(small)
        ctx = prob_context(small)
        ctx = ctx.with_columns(pl.col(pl.Float64).cast(pl.Float32))
        if use_anchor:
            from anchor import anchor_features
            anc = anchor_features(small, _target_recs("test", small["t"].unique(), maps, c), n_buckets=16)
            ctx = ctx.join(anc, on=["q", "t"], how="left")
            del anc
        del small
        cfn = f"{WORK}/ctx_test_{c}.parquet"
        ctx.write_parquet(cfn)
        del ctx
        gc.collect()
        # stage 2/3 in row slices, pulling only the matching context rows from disk
        scored = []
        for fn in fns:
            full = pl.read_parquet(f"{WORK}/{fn}")
            for i0 in range(0, full.height, 2_000_000):
                f = enrich(full.slice(i0, 2_000_000), nf)
                cx = pl.scan_parquet(cfn).filter(pl.col("q").is_in(f["q"].unique().implode())).collect()
                f = f.join(cx, on=["q", "t", "src"], how="left")
                scored.append(f.select("q", "t", "p1").with_columns(p=pl.Series(_pred(m2, f, feats2), dtype=pl.Float32)))
                del f, cx
                gc.collect()
            del full
            gc.collect()
        pl.concat(scored).write_parquet(out_fn)
        del scored
        gc.collect()
        print("scored", c, flush=True)
    sc = pl.concat([pl.read_parquet(f"{WORK}/{tag}_{c}.parquet") for c in countries])
    sc.write_parquet(f"{WORK}/test_scores.parquet")
    cfg = json.load(open(f"{WORK}/decode_cfg.json")) if os.path.exists(f"{WORK}/decode_cfg.json") else {"mode": "thr", "thr": 0.7}
    print("decode cfg", cfg)
    if cfg["mode"] == "thr":
        pred = sc.filter(pl.col("p") == pl.col("p").max().over("t")).filter(pl.col("p") > cfg["thr"]).select("q", "t")
    else:
        pred = decode(sc.with_columns(p=pl.col("p") ** cfg.get("pow", 1.0)).select("q", "t", "p"), exclusive=True)
    write_outputs(sc.select("q", "t"), pred.select("q", "t"))


def predict():
    import lightgbm as lgb
    from decode import decode
    m = lgb.Booster(model_file=f"{WORK}/lgb.txt")
    feats = json.load(open(f"{WORK}/feats.json"))
    scored = []
    nf = name_freq_tables("test")
    for fn in _files("test"):
        f = enrich(pl.read_parquet(f"{WORK}/{fn}"), nf)
        p = m.predict(f.select(feats).to_numpy().astype(np.float32), num_threads=os.cpu_count())
        scored.append(f.select("q", "t").with_columns(p=pl.Series(p, dtype=pl.Float32)))
        del f
        gc.collect()
    sc = pl.concat(scored)
    sc.write_parquet(f"{WORK}/test_scores.parquet")
    pred = decode(sc, exclusive=True)
    write_outputs(sc.select("q", "t"), pred.select("q", "t"))


def _write(long, colname, allq, path):
    agg = long.group_by("q").agg(pl.col("t").unique(maintain_order=True).str.join(",").alias(colname))
    out = pl.DataFrame({"source1_entity_id": allq}).join(
        agg.rename({"q": "source1_entity_id"}), on="source1_entity_id", how="left").fill_null("")
    with open(path, "w") as fh:
        fh.write(f"source1_entity_id\t{colname}\n")
        for a, b in out.iter_rows():
            fh.write(f"{a}\t{b}\n")


def write_outputs(cand, pred):
    os.makedirs(OUT, exist_ok=True)
    allq = pl.read_parquet(f"{WORK}/test_s1.parquet", columns=["id"])["id"].to_list()
    _write(cand, "candidate_entity_ids", allq, f"{OUT}/candidate_pairs.tsv")
    _write(pred, "matched_entity_ids", allq, f"{OUT}/matching_results.tsv")
    print("wrote", OUT, "matches", pred.height)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "candidates":
        candidates(sys.argv[2])
    elif cmd == "train":
        train()
    elif cmd == "evaluate":
        evaluate()
    elif cmd == "predict":
        predict()
    elif cmd == "train2":
        train2()
    elif cmd == "predict2":
        predict2()
    elif cmd == "train3":
        train3()
    elif cmd == "predict3":
        predict2(stage="3", tag="score3")
