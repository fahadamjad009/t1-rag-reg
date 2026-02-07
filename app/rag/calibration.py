from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CalibBand:
    # Raw score interval [lo, hi] -> calibrated interval [out_lo, out_hi]
    lo: float
    hi: float
    out_lo: float
    out_hi: float


# These bands are deliberately conservative.
# They can be tuned later using your eval_compare JSON outputs.
BANDS: dict[str, list[CalibBand]] = {
    # vector scores already tend to be in [0,1]
    "vector": [
        CalibBand(0.00, 0.30, 0.00, 0.20),
        CalibBand(0.30, 0.45, 0.20, 0.45),
        CalibBand(0.45, 0.60, 0.45, 0.70),
        CalibBand(0.60, 1.00, 0.70, 1.00),
    ],
    # bm25 scores are unbounded-ish; typical ranges depend on corpus
    "bm25": [
        CalibBand(0.00, 3.00, 0.00, 0.20),
        CalibBand(3.00, 6.00, 0.20, 0.45),
        CalibBand(6.00, 10.0, 0.45, 0.75),
        CalibBand(10.0, 30.0, 0.75, 1.00),
    ],
    # hybrid_rrf in your system is producing ~[0,1] (at least in your examples)
    # DO NOT saturate to 1.0 around 0.5 — keep it similar to vector but slightly friendlier.
    "hybrid_rrf": [
        CalibBand(0.00, 0.25, 0.00, 0.25),
        CalibBand(0.25, 0.45, 0.25, 0.55),
        CalibBand(0.45, 0.65, 0.55, 0.85),
        CalibBand(0.65, 1.00, 0.85, 1.00),
    ],
}


def _clamp01(x: float) -> float:
    if x != x:  # NaN
        return 0.0
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def calibrate_best(best_score: float | None, mode: str) -> float:
    """
    Convert retriever raw best_score -> calibrated confidence in [0,1]
    comparable across modes.
    """
    if best_score is None:
        return 0.0

    try:
        raw = float(best_score)
    except Exception:
        return 0.0

    m = (mode or "vector").strip().lower()
    bands = BANDS.get(m, BANDS["vector"])

    # Find the first band that contains raw; if outside, extrapolate by clamping to edges.
    for b in bands:
        if raw <= b.hi:
            # linear interpolate within [b.lo, b.hi]
            if b.hi <= b.lo:
                return _clamp01(b.out_lo)
            t = (raw - b.lo) / (b.hi - b.lo)
            out = b.out_lo + t * (b.out_hi - b.out_lo)
            return _clamp01(out)

    # raw above highest band -> return 1.0 (max confidence)
    return 1.0


def explain_calibration() -> dict:
    """
    Return the calibration bands so the API can expose them for debugging.
    """
    out: dict[str, list[dict]] = {}
    for mode, bands in BANDS.items():
        out[mode] = [
            {"raw": [b.lo, b.hi], "calib": [b.out_lo, b.out_hi]}
            for b in bands
        ]
    return out
