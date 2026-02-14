from __future__ import annotations

import re
from typing import List


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def rewrite_query(q: str, max_rewrites: int = 4) -> List[str]:
    """
    Deterministic query expansion for regulatory language.
    No LLM dependency.
    Designed to fix 'not_retrieved' failures.
    """

    q0 = _norm(q)
    if not q0:
        return []

    out: List[str] = [q0]
    ql = q0.lower()

    # ---- AUSTRAC customer identification ----
    if "customer identification" in ql or "identification procedures" in ql:
        out.append("AUSTRAC customer identification and verification procedures")
        out.append("customer identification and verification easy reference guide AUSTRAC")
        out.append("know your customer KYC requirements AUSTRAC")

    # ---- KYC ----
    if "kyc" in ql:
        out.append("customer identification know your customer AUSTRAC")
        out.append("AUSTRAC customer due diligence identification requirements")

    # ---- AML / CTF ----
    if "aml" in ql or "ctf" in ql:
        out.append("AML CTF Rules customer identification AUSTRAC")
        out.append("AUSTRAC AML CTF customer due diligence procedures")

    # ---- Deduplicate preserve order ----
    seen = set()
    final: List[str] = []
    for s in out:
        s = _norm(s)
        if not s or s in seen:
            continue
        seen.add(s)
        final.append(s)
        if len(final) >= max_rewrites:
            break

    return final
