"""
Novel suspicious-domain features for the LegitPhish dataset.

Every feature is computed from the raw URL string only, so it can be
reproduced on any URL dataset. Features are grouped into families so the
ablation study in run_experiments.py can measure each family separately.

Families
--------
IP      : URLs that use an IP address (incl. hex / decimal / IPv6) instead of a domain
LOGIN   : fake login / credential-harvesting patterns
SUBDOM  : misleading subdomains (brand or full domain hidden in the subdomain)
BRAND   : brand impersonation and typosquatting (e.g. "metamsk" vs "metamask")
HOSTING : abuse of free hosting / site-builder platforms and risky TLDs
OBFUSC  : obfuscation tricks (punycode, '@' userinfo, ports, shorteners, encoding)
RISK    : SDRS - an interpretable Suspicious Domain Risk Score built from the flags
"""

from __future__ import annotations

import ipaddress
import math
import re
from urllib.parse import unquote, urlsplit

import numpy as np
import pandas as pd
import tldextract

# Offline extractor (uses the public-suffix snapshot bundled with tldextract,
# so results are identical on every machine and need no internet).
_EXTRACT = tldextract.TLDExtract(suffix_list_urls=())

BRANDS = [
    "paypal", "apple", "icloud", "microsoft", "office", "outlook", "office365",
    "google", "gmail", "facebook", "instagram", "whatsapp", "netflix", "amazon",
    "ebay", "linkedin", "twitter", "dropbox", "adobe", "docusign", "yahoo",
    "metamask", "coinbase", "binance", "blockchain", "trustwallet", "opensea",
    "ledger", "chase", "wellsfargo", "bankofamerica", "citibank", "hsbc",
    "americanexpress", "visa", "mastercard", "dhl", "fedex", "ups", "usps",
    "steam", "roblox", "spotify", "telegram", "att", "verizon", "irs",
    "wetransfer", "sharepoint", "onedrive", "zoom", "discord",
]
# Short brands are only matched as whole tokens to avoid false hits ("ups" in "groups").
_SHORT_BRANDS = {b for b in BRANDS if len(b) <= 4}

LOGIN_WORDS = [
    "login", "log-in", "signin", "sign-in", "logon", "verify", "verification",
    "account", "update", "secure", "security", "banking", "confirm", "password",
    "passwd", "credential", "wallet", "webscr", "auth", "authenticate",
    "unlock", "suspend", "recover", "restore", "billing", "invoice", "validate",
]
PASSWORD_WORDS = ["password", "passwd", "pwd", "credential", "pin", "otp"]

FREE_HOSTING = [
    "weebly.com", "weeblysite.com", "wixsite.com", "godaddysites.com",
    "square.site", "vercel.app", "pages.dev", "workers.dev", "github.io",
    "gitbook.io", "firebaseapp.com", "web.app", "netlify.app", "glitch.me",
    "herokuapp.com", "000webhostapp.com", "blogspot.com", "sites.google.com",
    "dweb.link", "ipfs.io", "webflow.io", "framer.app", "replit.app",
    "repl.co", "wordpress.com", "myshopify.com", "carrd.co", "jimdosite.com",
    "yolasite.com", "site123.me", "mystrikingly.com", "azurewebsites.net",
    "appspot.com", "r2.dev", "onrender.com", "surge.sh", "translate.goog",
]
SHORTENERS = [
    "bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd", "buff.ly",
    "cutt.ly", "rebrand.ly", "shorturl.at", "rb.gy", "t.ly", "tiny.cc", "s.id",
]
RISKY_TLDS = {
    "tk", "ml", "ga", "cf", "gq", "xyz", "top", "shop", "icu", "buzz", "cyou",
    "click", "link", "live", "online", "site", "website", "rest", "monster",
    "sbs", "cfd", "lol", "work", "support", "help", "zip", "mov", "info",
}
TLD_LIKE_TOKENS = {"com", "net", "org", "gov", "edu", "co", "io", "info"}


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _is_ip_host(host: str) -> int:
    if not host:
        return 0
    h = host.strip("[]")
    try:
        ipaddress.ip_address(h)
        return 1
    except ValueError:
        pass
    if re.fullmatch(r"0x[0-9a-f]+", h) or re.fullmatch(r"\d{8,10}", h):  # hex / decimal IP
        return 1
    if re.fullmatch(r"(0x[0-9a-f]{1,2}|\d{1,3})(\.(0x[0-9a-f]{1,2}|\d{1,3})){3}", h):
        return 1
    return 0


def _tokens(s: str) -> list[str]:
    return [t for t in re.split(r"[^a-z0-9]+", s.lower()) if t]


def _brand_hits(text: str) -> set[str]:
    toks = set(_tokens(text))
    low = text.lower()
    hits = set()
    for b in BRANDS:
        if b in _SHORT_BRANDS:
            if b in toks:
                hits.add(b)
        elif b in low:
            hits.add(b)
    return hits


def _entropy(s: str) -> float:
    if not s:
        return 0.0
    counts = pd.Series(list(s)).value_counts().to_numpy()
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())


def extract_url_features(url: str) -> dict:
    raw = str(url).strip()
    parsed = urlsplit(raw if "://" in raw else "http://" + raw)
    try:
        host = (parsed.hostname or "").lower()
        port = parsed.port
    except ValueError:
        host, port = "", None
    path_q = unquote((parsed.path or "") + "?" + (parsed.query or "")).lower()

    ext = _EXTRACT(host)
    sub = ext.subdomain.lower()
    dom = ext.domain.lower()
    suffix = ext.suffix.lower()
    registered = f"{dom}.{suffix}" if suffix else dom
    sub_labels = [s for s in sub.split(".") if s and s != "www"]

    ip_host = _is_ip_host(host)

    # ---- LOGIN family
    full_low = unquote(raw).lower()
    login_count = sum(full_low.count(w) for w in LOGIN_WORDS)
    login_in_host = int(any(w in host for w in LOGIN_WORDS))
    login_in_path = int(any(w in path_q for w in LOGIN_WORDS))
    pwd_word = int(any(re.search(rf"(^|[^a-z]){w}([^a-z]|$)", path_q) for w in PASSWORD_WORDS))
    form_like = int(bool(re.search(r"\.(php|aspx?|jsp|cgi)(\?|$)|/wp-(admin|content|includes)/", path_q)))

    # ---- BRAND family
    brands_sub = _brand_hits(sub)
    brands_path = _brand_hits(path_q)
    brand_is_domain = dom in BRANDS  # the real brand's own domain (paypal.com, apple.com ...)
    brands_dom = _brand_hits(dom)
    brand_in_subdomain = int(bool(brands_sub) and not brand_is_domain)
    brand_in_path = int(bool(brands_path) and not brand_is_domain)
    brand_domain_mismatch = int(bool(brands_sub | brands_path | brands_dom) and not brand_is_domain)
    # typosquatting: domain label is 1-2 edits away from a brand but not equal to it
    # On free-hosting platforms the attacker-chosen name is the subdomain (x.weebly.com),
    # so it is checked as well. Labels are split on hyphens / digits ("metamsk01" -> "metamsk").
    free_host_pre = any(host == f or host.endswith("." + f) for f in FREE_HOSTING)
    cand = [dom] + (sub_labels if free_host_pre else [])
    toks = {t for c in cand for t in re.split(r"[-0-9_]+", c) if len(t) >= 5}
    typosquat = int(any(0 < _levenshtein(t, b) <= (1 if len(b) <= 6 else 2)
                        for t in toks for b in BRANDS if len(b) >= 5 and t not in BRANDS))
    # brand combined with extra words in the registered domain, e.g. "officalmetamask", "paypal-secure"
    brand_combo_domain = int(bool(brands_dom) and not brand_is_domain)

    # ---- SUBDOM family (misleading subdomains)
    subdomain_depth = len(sub_labels)
    tld_token_in_sub = int(any(lbl in TLD_LIKE_TOKENS for lbl in sub_labels))
    domain_in_sub = int(bool(re.search(r"[a-z0-9-]+\.(com|net|org)(\.|-|$)", sub)))
    longest_sub_label = max((len(s) for s in sub_labels), default=0)
    sub_entropy = _entropy(sub.replace(".", ""))
    random_sub = int(longest_sub_label >= 20 or (sub_entropy > 3.8 and longest_sub_label >= 12))

    # ---- HOSTING family
    free_host = int(any(host == f or host.endswith("." + f) for f in FREE_HOSTING) or "ipfs" in host)
    risky_tld = int(suffix.split(".")[-1] in RISKY_TLDS) if suffix else 0

    # ---- OBFUSC family
    punycode = int("xn--" in host)
    at_sym = int("@" in parsed.netloc)
    nonstd_port = int(port is not None and port not in (80, 443))
    shortener = int(registered in SHORTENERS)
    pct_encoded = raw.count("%")
    double_slash_redirect = int("//" in (parsed.path or ""))
    host_digits = sum(c.isdigit() for c in host)
    host_digit_ratio = host_digits / len(host) if host else 0.0
    host_hyphens = host.count("-")

    credential_combo = int((login_in_host or login_in_path) and (brand_domain_mismatch or free_host or ip_host or typosquat))

    feats = {
        # IP
        "ip_literal_host": ip_host,
        # LOGIN
        "login_keyword_count": login_count,
        "login_in_host_flag": login_in_host,
        "login_in_path_flag": login_in_path,
        "path_has_password_word": pwd_word,
        "server_script_form_flag": form_like,
        # SUBDOM
        "subdomain_depth": subdomain_depth,
        "tld_token_in_subdomain": tld_token_in_sub,
        "full_domain_in_subdomain": domain_in_sub,
        "longest_subdomain_label": longest_sub_label,
        "subdomain_entropy": round(sub_entropy, 6),
        "random_looking_subdomain": random_sub,
        # BRAND
        "brand_in_subdomain_flag": brand_in_subdomain,
        "brand_in_path_flag": brand_in_path,
        "brand_domain_mismatch_flag": brand_domain_mismatch,
        "brand_combo_domain_flag": brand_combo_domain,
        "typosquat_flag": typosquat,
        # HOSTING
        "free_hosting_flag": free_host,
        "risky_tld_flag": risky_tld,
        # OBFUSC
        "punycode_flag": punycode,
        "userinfo_at_symbol_flag": at_sym,
        "nonstandard_port_flag": nonstd_port,
        "url_shortener_flag": shortener,
        "percent_encoding_count": pct_encoded,
        "path_double_slash_flag": double_slash_redirect,
        "host_digit_ratio": round(host_digit_ratio, 6),
        "host_hyphen_count": host_hyphens,
        "credential_harvest_combo": credential_combo,
    }
    return feats


FEATURE_FAMILIES = {
    "IP": ["ip_literal_host"],
    "LOGIN": ["login_keyword_count", "login_in_host_flag", "login_in_path_flag",
              "path_has_password_word", "server_script_form_flag", "credential_harvest_combo"],
    "SUBDOM": ["subdomain_depth", "tld_token_in_subdomain", "full_domain_in_subdomain",
               "longest_subdomain_label", "subdomain_entropy", "random_looking_subdomain"],
    "BRAND": ["brand_in_subdomain_flag", "brand_in_path_flag", "brand_domain_mismatch_flag",
              "brand_combo_domain_flag", "typosquat_flag"],
    "HOSTING": ["free_hosting_flag", "risky_tld_flag"],
    "OBFUSC": ["punycode_flag", "userinfo_at_symbol_flag", "nonstandard_port_flag",
               "url_shortener_flag", "percent_encoding_count", "path_double_slash_flag",
               "host_digit_ratio", "host_hyphen_count"],
    "RISK": ["sdrs_score"],
}

# Weights of the interpretable Suspicious Domain Risk Score (0-100).
SDRS_WEIGHTS = {
    "ip_literal_host": 20,
    "credential_harvest_combo": 15,
    "brand_domain_mismatch_flag": 12,
    "typosquat_flag": 12,
    "brand_in_subdomain_flag": 8,
    "free_hosting_flag": 8,
    "login_in_host_flag": 6,
    "login_in_path_flag": 4,
    "full_domain_in_subdomain": 5,
    "tld_token_in_subdomain": 3,
    "random_looking_subdomain": 3,
    "punycode_flag": 6,
    "userinfo_at_symbol_flag": 6,
    "risky_tld_flag": 4,
    "url_shortener_flag": 3,
    "nonstandard_port_flag": 3,
}


def add_novel_features(df: pd.DataFrame, url_col: str = "URL") -> pd.DataFrame:
    feats = pd.DataFrame([extract_url_features(u) for u in df[url_col]], index=df.index)
    total = sum(SDRS_WEIGHTS.values())
    feats["sdrs_score"] = (sum(feats[c] * w for c, w in SDRS_WEIGHTS.items()) / total * 100).round(3)
    return pd.concat([df, feats], axis=1)


def registered_domain(url: str) -> str:
    """Used for leakage-aware (domain-grouped) splitting."""
    raw = str(url).strip()
    try:
        host = (urlsplit(raw if "://" in raw else "http://" + raw).hostname or "").lower()
    except ValueError:
        host = ""
    if _is_ip_host(host):
        return host
    ext = _EXTRACT(host)
    # For free-hosting platforms each user site is a separate owner, so keep the full host.
    if any(host.endswith("." + f) for f in FREE_HOSTING):
        return host
    return f"{ext.domain}.{ext.suffix}" if ext.suffix else host


NOVEL_FEATURES = [c for fam in FEATURE_FAMILIES.values() for c in fam]
