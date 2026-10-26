"""Synthetic, offline as-of membership tests. No actual TSE listing claims."""

from copy import deepcopy

import pytest

from research.historical_universe import VERSION, asof_historical_universe

CODE = "12345678901234567"
HOLDOUT = "2025-03-21"


def decision(day="2024-02-05", at="2024-02-05T12:30:00+03:30", code=CODE):
    return dict(instrument_id=code, date=day, decision_at=at)


def record(**kwargs):
    value = dict(
        instrument_id=CODE, date="2024-02-05",
        published_at="2024-02-05T11:00:00+03:30",
        first_usable_at="2024-02-05T11:01:00+03:30",
        source_record_id="native-1", source_sha256="a" * 64,
        instrument_type_code=300, listed=True, trading_at_decision=True,
        identity_verified=True, effective_date_verified=True,
        source_publication_verified=True, historical_access_verified=True,
        market_classification_verified=True, listing_verified=True,
        decision_time_trading_verified=True,
    )
    value.update(kwargs)
    return value


def run(evidence=None, decisions=None):
    return asof_historical_universe(
        [decision()] if decisions is None else decisions,
        [record()] if evidence is None else evidence,
        holdout_start=HOLDOUT,
    )


def test_verified_ordinary_share_may_enter_research_census_but_not_signals():
    x = run()[0]
    assert x["status"] == "verified_eligible" and x["eligible_for_study"] is True
    assert x["version"] == VERSION and x["source_sha256"] == "a" * 64
    assert x["historical_membership_verified"] is True
    assert x["trading_eligibility_verified"] is True
    assert not x["accepted_for_signals"] and not x["executable_return_verified"]


def test_absent_data_never_turns_into_definite_nonmember():
    x = run([])[0]
    assert x["status"] == "no_verified_publication_asof"
    assert x["eligible_for_study"] is None
    assert not x["historical_membership_verified"]


@pytest.mark.parametrize("updates", [
    {"instrument_type_code": 400},
    {"listed": False, "trading_at_decision": False},
    {"trading_at_decision": False},
])
def test_verified_explicit_ineligibility_is_not_unknown(updates):
    x = run([record(**updates)])[0]
    assert x["status"] == "verified_ineligible"
    assert x["eligible_for_study"] is False


@pytest.mark.parametrize("flag", [
    "identity_verified", "effective_date_verified", "historical_access_verified",
    "market_classification_verified", "listing_verified", "decision_time_trading_verified",
])
def test_partial_attestation_never_produces_positive_or_negative_membership(flag):
    x = run([record(**{flag: False})])[0]
    assert x["status"] == "latest_evidence_unverified_asof"
    assert x["eligible_for_study"] is None


def test_future_publication_is_completely_invisible_to_earlier_decision():
    original = run([])
    extra = record(published_at="2024-02-05T15:00:00+03:30",
                   first_usable_at="2024-02-05T15:01:00+03:30")
    assert run([extra]) == original
    assert run([extra, record(source_record_id="native-2",
                              published_at="2024-02-06T15:00:00+03:30",
                              first_usable_at="2024-02-06T15:01:00+03:30")]) == original


def test_exact_publication_and_usable_boundaries_are_not_included():
    boundary = "2024-02-05T12:30:00+03:30"
    assert run([record(published_at=boundary, first_usable_at=boundary)]) == run([])
    assert run([record(first_usable_at=boundary)])[0]["eligible_for_study"] is None


def test_known_newer_publication_masks_older_attestation_until_usable():
    old = record(source_record_id="old", source_sha256="a" * 64)
    newer = record(source_record_id="new", source_sha256="b" * 64,
                   published_at="2024-02-05T12:00:00+03:30",
                   first_usable_at="2024-02-05T16:00:00+03:30")
    expected = run([old, newer])[0]
    assert expected["status"] == "latest_evidence_unverified_asof"
    assert expected["eligible_for_study"] is None and expected["source_sha256"] is None
    assert run([newer, old])[0] == expected


def test_known_newer_unverified_revision_cannot_leave_stale_positive():
    old, newer = record(source_record_id="old"), record(
        source_record_id="new", published_at="2024-02-05T12:00:00+03:30",
        first_usable_at="2024-02-05T12:01:00+03:30", listing_verified=False,
    )
    assert run([old, newer])[0]["status"] == "latest_evidence_unverified_asof"


def test_same_time_competitors_fail_closed_without_arbitrary_source_order():
    first = record(source_record_id="one")
    second = record(source_record_id="two", source_sha256="b" * 64,
                    listed=False, trading_at_decision=False)
    expected = run([first, second])[0]
    assert expected["status"] == "conflicting_latest_publications"
    assert expected["eligible_for_study"] is None
    assert run([second, first])[0] == expected


def test_source_with_unverified_publication_is_not_retrospectively_visible():
    assert run([record(source_publication_verified=False)])[0] == run([])[0]


def test_date_specific_attestation_does_not_prove_membership_on_another_date():
    old = record(date="2024-02-04", published_at="2024-02-04T10:00:00+03:30",
                 first_usable_at="2024-02-04T10:01:00+03:30")
    result = run([old])[0]
    assert result["status"] == "no_verified_publication_asof"
    assert result["eligible_for_study"] is None


def test_utc_timezone_equivalence_does_not_change_eligibility():
    x = run([record(published_at="2024-02-05T07:30:00Z",
                    first_usable_at="2024-02-05T07:31:00+00:00")])[0]
    assert x["eligible_for_study"] is True


def test_all_anchor_rows_kept_even_with_unknown_and_ineligible():
    next_code = "98765432101234567"
    decisions = [decision(), decision(code=next_code)]
    result = run([record(trading_at_decision=False)], decisions=decisions)
    assert [x["eligible_for_study"] for x in result] == [False, None]


@pytest.mark.parametrize("corruption", [
    {"date": HOLDOUT},
    {"source_sha256": "not-a-hash"},
    {"instrument_id": "۱۲۳۴۵۶۷۸۹۰"},
    {"instrument_type_code": True},
    {"listed": None},
    {"first_usable_at": "2024-02-05T10:00:00+03:30"},
    {"published_at": "2024-02-05T11:00:00"},
    {"trading_at_decision": True, "listed": False},
])
def test_invalid_evidence_is_rejected_without_coercion(corruption):
    with pytest.raises(ValueError):
        run([record(**corruption)])


def test_duplicate_source_id_and_anchor_are_rejected():
    with pytest.raises(ValueError, match="Duplicate source"):
        run([record(), deepcopy(record())])
    with pytest.raises(ValueError, match="Duplicate instrument"):
        run(decisions=[decision(), decision()])


def test_mismatched_tehran_date_and_sealed_decision_rejected():
    with pytest.raises(ValueError):
        run(decisions=[decision(day="2024-02-06")])
    with pytest.raises(ValueError):
        run(decisions=[decision(day=HOLDOUT, at="2025-03-21T12:30:00+03:30")])


@pytest.mark.parametrize("published_at,first_usable_at", [
    ("2025-03-21T00:00:00+03:30", "2025-03-21T00:01:00+03:30"),
    ("2024-02-05T11:00:00+03:30", "2025-03-21T00:01:00+03:30"),
    ("2025-03-20T21:00:00Z", "2025-03-20T21:01:00Z"),
])
def test_sealed_publication_or_availability_rejected_even_for_preholdout_effective_date(
    published_at, first_usable_at,
):
    with pytest.raises(ValueError, match="Sealed-period"):
        run([record(published_at=published_at, first_usable_at=first_usable_at)])


def test_sealed_evidence_is_rejected_before_reading_status_fields():
    class ProtectedEvidence(dict):
        def __getitem__(self, key):
            if key in {"instrument_type_code", "listed", "trading_at_decision"}:
                pytest.fail("Sealed status was accessed before the cutoff check")
            return super().__getitem__(key)

    poisoned = ProtectedEvidence(record(
        published_at="2025-03-21T00:00:00+03:30",
        first_usable_at="2025-03-21T00:00:01+03:30",
    ))
    with pytest.raises(ValueError, match="Sealed-period"):
        run([poisoned])


@pytest.mark.parametrize("code", ["9", "9" * 7, "9" * 21, "9" * 30])
def test_source_native_instrument_id_bounds_match_instrument_history(code):
    """Instrument-history accepts 1..30 ASCII digits, with first digit nonzero."""
    result = run([record(instrument_id=code)], decisions=[decision(code=code)])[0]
    assert result["instrument_id"] == code
    assert result["status"] == "verified_eligible"


@pytest.mark.parametrize("bad_code", ["0", "0" + "9" * 16, "9" * 31, 9, True, "۹" * 17])
def test_invalid_source_native_instrument_ids_remain_rejected(bad_code):
    with pytest.raises(ValueError, match="Stable ASCII numeric instrument ID"):
        run(decisions=[decision(code=bad_code)])
