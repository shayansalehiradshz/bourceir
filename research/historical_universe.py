"""Fail-closed, research-only as-of audit for historically eligible instruments.

Evidence is independently attested upstream. This module neither authenticates
source bytes nor infers tradability from TSETMC state codes or missing candles.
"""

import re
from collections import defaultdict
from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

VERSION = "historical-universe-asof-v1"
_TEHRAN = ZoneInfo("Asia/Tehran")
_INSTRUMENT = re.compile(r"[1-9][0-9]{7,19}\Z")
_SHA256 = re.compile(r"[a-f0-9]{64}\Z")
_TIME = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(?:Z|[+-][0-9]{2}:[0-9]{2})\Z")
_PROOFS = (
    "identity_verified",
    "effective_date_verified",
    "source_publication_verified",
    "historical_access_verified",
    "market_classification_verified",
    "listing_verified",
    "decision_time_trading_verified",
)
_REQUIRED = {
    "instrument_id", "date", "published_at", "first_usable_at", "source_sha256",
    "source_record_id", "instrument_type_code", "listed", "trading_at_decision",
    *_PROOFS,
}


def _day(value):
    if type(value) is not str:
        raise ValueError("A canonical ISO date is required")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError("A canonical ISO date is required") from exc
    if parsed.isoformat() != value:
        raise ValueError("A canonical ISO date is required")
    return value


def _stamp(value):
    if type(value) is not str or _TIME.fullmatch(value) is None:
        raise ValueError("Explicit offset-aware ISO timestamps with seconds are required")
    try:
        stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("Invalid timestamp") from exc
    if stamp.tzinfo is None or stamp.utcoffset() is None:
        raise ValueError("Timezone is required")
    return stamp.astimezone(timezone.utc)


def _instrument(value):
    if type(value) is not str or _INSTRUMENT.fullmatch(value) is None:
        raise ValueError("Stable ASCII numeric instrument ID is required")
    return value


def _validate_evidence(row, cutoff):
    if not isinstance(row, dict) or not _REQUIRED <= set(row):
        raise ValueError("Complete dated universe evidence is required")
    # Reject sealed-period records before accessing their status/classification.
    day = _day(row["date"])
    if day >= cutoff:
        raise ValueError("Sealed-period evidence cannot enter research")
    code = _instrument(row["instrument_id"])
    published = _stamp(row["published_at"])
    usable = None if row["first_usable_at"] is None else _stamp(row["first_usable_at"])
    if usable is not None and usable < published:
        raise ValueError("Evidence cannot be usable before its publication")
    if type(row["source_record_id"]) is not str or not row["source_record_id"].strip():
        raise ValueError("A source-native record ID is required")
    if type(row["source_sha256"]) is not str or _SHA256.fullmatch(row["source_sha256"]) is None:
        raise ValueError("Source byte digest must be SHA-256")
    if type(row["instrument_type_code"]) is not int or not 0 < row["instrument_type_code"] < 10_000:
        raise ValueError("Source market-type code must be an explicit integer")
    for name in (*_PROOFS, "listed", "trading_at_decision"):
        if type(row[name]) is not bool:
            raise ValueError("Evidence states and attestations must be explicit booleans")
    if not row["listed"] and row["trading_at_decision"]:
        raise ValueError("An unlisted instrument cannot have verified trade eligibility")
    return code, day, published, usable


def asof_historical_universe(decisions, evidence, *, holdout_start):
    """Keep every decision, admitting only externally attested same-day eligibility.

    Returns nullable `eligible_for_study` (never infer False from missing bars).
    Neither symbol names nor current-universe snapshots are used. Historical
    publication is a distinct gate from historical usability; later evidence
    must not affect earlier decisions. Exact-day attestations are NOT carried
    forward to another date without fresh proof.
    """
    cutoff = _day(holdout_start)
    if not isinstance(decisions, list) or not isinstance(evidence, list):
        raise ValueError("Decisions and evidence must be lists")
    anchors = []
    seen_decisions = set()
    for row in decisions:
        if not isinstance(row, dict) or not {"instrument_id", "date", "decision_at"} <= set(row):
            raise ValueError("Explicit historical decisions are required")
        code, day = _instrument(row["instrument_id"]), _day(row["date"])
        when = _stamp(row["decision_at"])
        if day >= cutoff or when.astimezone(_TEHRAN).date().isoformat() != day:
            raise ValueError("Decisions must match their Tehran date and precede holdout")
        if (code, day) in seen_decisions:
            raise ValueError("Duplicate instrument/day decision")
        seen_decisions.add((code, day))
        anchors.append((code, day, when))

    by_day = defaultdict(list)
    seen_sources = set()
    for row in evidence:
        code, day, published, usable = _validate_evidence(row, cutoff)
        unique = (code, day, row["source_record_id"])
        if unique in seen_sources:
            raise ValueError("Duplicate source record ID")
        seen_sources.add(unique)
        # An unverified publication time cannot be used as historical visibility.
        if row["source_publication_verified"]:
            by_day[code, day].append((published, usable, row))

    results = []
    for code, day, when in anchors:
        visible = [(published, usable, row)
                   for published, usable, row in by_day.get((code, day), ())
                   if published < when]
        result = dict(
            instrument_id=code, date=day, status="no_verified_publication_asof",
            eligible_for_study=None, source_sha256=None,
            historical_membership_verified=False, trading_eligibility_verified=False,
            accepted_for_signals=False, executable_return_verified=False, version=VERSION,
        )
        if visible:
            latest = max(item[0] for item in visible)
            candidates = [(usable, row) for published, usable, row in visible if published == latest]
            if len(candidates) != 1:
                result["status"] = "conflicting_latest_publications"
            else:
                usable, row = candidates[0]
                if usable is None or usable >= when or not all(row[k] for k in _PROOFS):
                    result["status"] = "latest_evidence_unverified_asof"
                else:
                    result["status"] = "verified_eligible" if (
                        row["instrument_type_code"] in (300, 303)
                        and row["listed"] and row["trading_at_decision"]
                    ) else "verified_ineligible"
                    result["eligible_for_study"] = result["status"] == "verified_eligible"
                    result["source_sha256"] = row["source_sha256"]
                    result["historical_membership_verified"] = True
                    result["trading_eligibility_verified"] = True
        results.append(result)
    return results
