"""Step 11: label-shift correction for nearby-house-number decoys.

Test contains ~4x more "sibling" decoys (same street, house number off by <=50) than
the labelled validation split, so the stage-3 model over-predicts them on test. For
each country we bucket candidate pairs by house-number offset (d10/d50/d500/dbig) x
whether the entity is already anchored at its own number, compare the per-entity rate
of decoded pairs in each bucket on test vs validation, and scale p by
min(1, valid_rate / test_rate) (3 fixed-point iterations), then re-decode.
Uses only test *predictions* (no test labels) and train labels. Unseen countries use
the US reference rates.
"""
import os, sys, json; sys.path.insert(0, os.path.dirname(__file__))
import polars as pl
import pipeline as P
from decode import decode
def firstnum(split):
    return pl.concat([pl.read_parquet(f"{P.WORK}/{split}_s{s}.parquet",columns=["id","nums"]) for s in "123"]).select("id",n=pl.col("nums").str.split(" ").list.first().replace("",None))
def add_cat(d, fn, pred_for_anchor):
    d=d.join(fn.rename({"id":"q","n":"qn"}),on="q",how="left").join(fn.rename({"id":"t","n":"tn"}),on="t",how="left")
    eq=(pl.col("qn")==pl.col("tn")).fill_null(False)
    # anchor: q already has a confident (p>0.5) candidate at its own number
    d=d.with_columns(eq=eq).with_columns(anch=((pl.col("eq")&(pl.col("p")>0.5)).sum().over("q")>0))
    diff=(pl.col("qn").is_not_null()&pl.col("tn").is_not_null()&~pl.col("eq")
          &~pl.col("qn").str.contains(pl.col("tn"),literal=True)&~pl.col("tn").str.contains(pl.col("qn"),literal=True))
    dd=(pl.col("qn").cast(pl.Int64,strict=False)-pl.col("tn").cast(pl.Int64,strict=False)).abs()
    return d.with_columns(cat=pl.when(~diff).then(pl.lit("base")).when(dd<=10).then(pl.lit("d10")).when(dd<=50).then(pl.lit("d50"))
        .when(dd<=500).then(pl.lit("d500")).otherwise(pl.lit("dbig"))+"_"+pl.col("anch").cast(pl.Utf8)).drop("qn","tn","eq","anch")
def rates(d, nq):
    pr=decode(d.select("q","t","p")).join(d.select("q","t","cat"),on=["q","t"])
    return {c:n/nq for c,n in pr.group_by("cat").len().rows()}
va=pl.read_parquet(P.WORK+"/valid_pred.parquet",columns=["q","t","p"])
s1=pl.read_parquet(P.WORK+"/train_s1.parquet",columns=["id","country"])
fn=firstnum("train"); ref={}
for c in ["US","India"]:
    ids=s1.filter(pl.col("country")==c)["id"]
    d=add_cat(va.filter(pl.col("q").is_in(ids.implode())),fn,None)
    ref[c]=rates(d, d["q"].n_unique())
print("ref",json.dumps({c:{k:round(v,4) for k,v in r.items()} for c,r in ref.items()}),flush=True)
fn=firstnum("test"); preds=[]; cfg={}
s1t=pl.read_parquet(P.WORK+"/test_s1.parquet",columns=["id","country"])
for c in sorted(s1t["country"].unique().to_list()):
    nqc=s1t.filter(pl.col("country")==c).height
    sc=pl.read_parquet(f"{P.WORK}/score3_{c}.parquet",columns=["q","t","p"])
    d=add_cat(sc,fn,None); qn=sc["q"].n_unique()
    for it in range(3):  # iterate: rates depend on the adjusted decode
        r=rates(d,qn)
        w={k:min(1.0,ref.get(c,ref["US"]).get(k,0)/v) if v>0 else 1.0 for k,v in r.items() if not k.startswith("base")}
        if it==0: cfg[c]=w; base_p=d["p"]
        else: cfg[c]={k:cfg[c].get(k,1.0)*w.get(k,1.0) for k in set(cfg[c])|set(w)}
        d=d.with_columns(p=base_p*pl.col("cat").replace_strict(cfg[c],default=1.0,return_dtype=pl.Float32))
    pr=decode(d.select("q","t","p"))
    print(c,"weights",{k:round(v,3) for k,v in sorted(cfg[c].items())},"mean k",round(pr.height/nqc,4),flush=True)
    preds.append(pr.select("q","t")); del sc,d
json.dump(cfg,open(P.WORK+"/shift_cfg.json","w"))
os.makedirs(P.OUT,exist_ok=True)
allq=pl.read_parquet(P.WORK+"/test_s1.parquet",columns=["id"])["id"].to_list()
P._write(pl.concat(preds),"matched_entity_ids",allq,P.OUT+"/matching_results.tsv")
