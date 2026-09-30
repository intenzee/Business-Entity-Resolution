"""Country-agnostic normalisation of business names and addresses.

Every rule here is generic (Unicode folding, punctuation, legal-form detection,
street-type canonicalisation). Nothing branches on a specific country value, so
an unseen label (e.g. France) goes through exactly the same code path.
"""
import re
import unicodedata
from anyascii import anyascii

# Legal forms: kept out of the "core" name and exposed as a separate signal.
LEGAL = {
    "inc", "incorporated", "llc", "llp", "lp", "ltd", "limited", "pvt", "private", "pvtltd",
    "corp", "corporation", "co", "company", "pc", "plc", "pllc", "pa", "lc", "llc.", "ltda",
    "sarl", "sas", "sasu", "sa", "eurl", "snc", "sci", "ei", "eirl", "scop", "scm", "selarl",
    "gmbh", "ag", "bv", "nv", "the", "of", "and", "et", "de", "du", "des", "la", "le", "les",
    "mr", "sri", "shri", "dba", "d", "b", "a", "l", "c", "s", "p",
}
# Tokens that are pure legal-form markers (for the legal-form agreement feature).
LEGAL_FORM = {
    "inc", "incorporated", "llc", "llp", "lp", "ltd", "limited", "pvt", "private", "corp",
    "corporation", "co", "company", "pc", "plc", "pllc", "sarl", "sas", "sasu", "sa", "eurl",
    "snc", "sci", "ei", "eirl",
}
LEGAL_CANON = {"incorporated": "inc", "limited": "ltd", "private": "pvt", "corporation": "corp",
               "company": "co"}

# Street / address type canonicalisation (abbreviation -> canonical).  Seed list of
# generic postal abbreviations; the data-mined dictionary (see mine.py) extends it.
ADDR_CANON = {
    "rd": "road", "st": "street", "str": "street", "ave": "avenue", "av": "avenue", "avn": "avenue",
    "dr": "drive", "drv": "drive", "cir": "circle", "ct": "court", "crt": "court", "ln": "lane",
    "blvd": "boulevard", "bd": "boulevard", "bld": "boulevard", "boul": "boulevard",
    "pl": "place", "hwy": "highway", "hy": "highway", "pkwy": "parkway", "trl": "trail",
    "ter": "terrace", "sq": "square", "mt": "mount", "ft": "fort", "apt": "apartment",
    "appt": "apartment", "ste": "suite", "fl": "floor", "flr": "floor", "bldg": "building",
    "no": "", "nº": "", "n°": "", "hno": "", "h": "", "unit": "", "pmb": "pmb", "po": "po",
    "r": "rue", "imp": "impasse", "all": "allee", "chem": "chemin", "rte": "route",
    "n": "north", "s": "south", "e": "east", "w": "west", "ne": "northeast", "nw": "northwest",
    "se": "southeast", "sw": "southwest", "nr": "near", "opp": "opposite", "extn": "extension",
    "ext": "extension", "sec": "sector", "rp": "rondpoint", "crs": "cross", "mg": "marg",
    "st.": "street",
}
LANDMARK = {"near", "opposite", "behind", "beside", "next", "cote", "face", "pres", "ke", "samne", "pas"}

_ws = re.compile(r"\s+")
_nonalnum = re.compile(r"[^a-z0-9 ]+")
_num = re.compile(r"\d+")
_url = re.compile(r"(^|\s)(https?://)?(www\.)?([a-z0-9\-]+)\.(com|net|org|in|co|io|fr|us|biz|info)\b")


def fold(s: str) -> str:
    """Unicode -> ASCII lower-case (handles accents and Indic scripts)."""
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = anyascii(s).lower()
    return s


def name_tokens(raw: str, translit: dict | None = None):
    """Return (all_tokens, core_tokens, legal_forms) for a business name."""
    s = fold(raw)
    s = s.replace("&", " and ").replace("+", " and ")
    s = _url.sub(lambda m: " " + m.group(4) + " ", s)
    s = s.replace("#", " ")
    s = s.replace("-", " ").replace("/", " ").replace("'", "")
    s = re.sub(r"\b([a-z])\.(?=[a-z]\.)", r"\1", s)  # l.l.c. -> llc
    s = _nonalnum.sub(" ", s)
    toks = [t for t in _ws.split(s) if t]
    if translit:
        toks = [translit.get(t, t) for t in toks]
    legal = sorted({LEGAL_CANON.get(t, t) for t in toks if t in LEGAL_FORM})
    core = [t for t in toks if t not in LEGAL]
    if not core:
        core = toks
    return toks, core, legal


def addr_tokens(raw: str, canon: dict | None = None):
    """Return (tokens, numbers, landmark_flag) for an address."""
    if not raw or raw == "<NULL>":
        return [], [], 0
    s = re.sub(r"\b[nN]\s?[°º]", " ", raw).replace("°", " ").replace("º", " ").replace("#", " ")
    s = fold(s).replace("<null>", " ").replace("n/a", " ")
    s = s.replace("'", " ").replace("’", " ")
    s = re.sub(r"(\d)(st|nd|rd|th|eme|er|e)\b", r"\1", s)  # 1st -> 1
    s = _nonalnum.sub(" ", s.replace("-", " ").replace("/", " "))
    toks = []
    cmap = canon if canon is not None else ADDR_CANON
    lm = 0
    for t in _ws.split(s):
        if not t:
            continue
        if t.isdigit():
            t = t.lstrip("0") or "0"
            toks.append(t)
            continue
        c = cmap.get(t, t)
        if c in LANDMARK:
            lm = 1
        if c:
            toks.append(c)
    nums = [t for t in toks if t.isdigit()]
    return toks, nums, lm
