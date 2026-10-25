"""Offline synthetic evidence only. No live market data, credentials or network."""

from copy import deepcopy

import pytest

from research.pit_fundamental_features import VERSION, asof_verified_monthly_revenue

CODE = '12345678901234567'
SHA1 = 'a' * 64
SHA2 = 'b' * 64
CUTOFF = '2025-03-21'


def decision(day='2024-02-05', at='2024-02-05T12:30:00+03:30'):
    return dict(instrument_id=CODE, date=day, decision_at=at)


def disclosure(**changes):
    row = dict(instrument_id=CODE, period_end='2024-01-31',
               published_at='2024-02-04T18:00:00+03:30', first_usable_at='2024-02-04T18:01:00+03:30',
               source_sha256=SHA1, tracing_no='12345', amount_rial='500000000',
               identity_verified=True, value_verified=True, historical_availability_verified=True,
               template_reconciled=True, report_kind='monthly_revenue', period_months=1,
               is_consolidated=False, currency='IRR')
    row.update(changes)
    return row


def run(decisions=None, filings=None):
    return asof_verified_monthly_revenue(
        [decision()] if decisions is None else decisions,
        [disclosure()] if filings is None else filings,
        holdout_start=CUTOFF,
    )


def test_verified_past_publication_is_available_and_not_a_signal():
    row = run()[0]
    assert row['status'] == 'verified_as_of'
    assert row['period_end'] == '2024-01-31'
    assert row['amount_rial'] == '500000000'
    assert row['source_sha256'] == SHA1
    assert row['version'] == VERSION
    assert row['accepted_for_signals'] is False and row['probability_calibrated'] is False


@pytest.mark.parametrize('changes', [
    dict(historical_availability_verified=False),
    dict(identity_verified=False),
    dict(value_verified=False),
])
def test_unverified_documents_cannot_be_promoted_to_model_features(changes):
    row = run(filings=[disclosure(**changes)])[0]
    assert row['status'] == 'historical_evidence_unverified' and row['amount_rial'] is None


def test_publication_after_market_close_stays_unknown_until_next_decision():
    row = disclosure(published_at='2024-02-05T13:00:00+03:30',
                     first_usable_at='2024-02-05T13:01:00+03:30')
    before, after = run(decisions=[decision(), decision('2024-02-06', '2024-02-06T12:30:00+03:30')],
                        filings=[row])
    assert before['status'] == 'not_yet_available' and before['amount_rial'] is None
    assert after['status'] == 'verified_as_of' and after['amount_rial'] == '500000000'


def test_exact_time_not_admitted_and_timezone_comparisons_are_absolute():
    first = disclosure(published_at='2024-02-05T09:00:00+00:00',
                       first_usable_at='2024-02-05T09:00:00+00:00')
    row = run(filings=[first])[0]
    assert row['status'] == 'not_yet_available'
    first['first_usable_at'] = '2024-02-05T08:59:59+00:00'
    first['published_at'] = '2024-02-05T08:59:00+00:00'
    assert run(filings=[first])[0]['status'] == 'verified_as_of'


def test_later_revision_never_rewrites_earlier_decision():
    initial = disclosure()
    restatement = disclosure(tracing_no='67890', source_sha256=SHA2,
                             amount_rial='700000000', published_at='2024-02-08T10:00:00+03:30',
                             first_usable_at='2024-02-08T10:01:00+03:30')
    before, after = run(
        decisions=[decision(), decision('2024-02-09', '2024-02-09T12:30:00+03:30')],
        filings=[restatement, initial],
    )
    assert before['amount_rial'] == '500000000' and before['source_sha256'] == SHA1
    assert after['amount_rial'] == '700000000' and after['source_sha256'] == SHA2


def test_latest_fiscal_period_takes_precedence_over_recent_old_period_revision():
    latest = disclosure(period_end='2024-02-02', published_at='2024-02-03T11:00:00+03:30',
                        first_usable_at='2024-02-03T11:01:00+03:30')
    older_revision = disclosure(period_end='2024-01-31', tracing_no='67890', source_sha256=SHA2,
                                published_at='2024-02-04T10:00:00+03:30',
                                first_usable_at='2024-02-04T10:01:00+03:30')
    assert run(filings=[older_revision, latest])[0]['period_end'] == '2024-02-02'


@pytest.mark.parametrize('field,value', [
    ('amount_rial', 100), ('amount_rial', '-1'), ('amount_rial', '1.5'),
    ('source_sha256', 'not-a-hash'), ('historical_availability_verified', 'true'),
    ('tracing_no', 123), ('first_usable_at', '2024-02-01T00:00:00+03:30'),
    ('published_at', '2024-03-21T18:00:00'), ('period_end', '2024-03-01'),
])
def test_invalid_evidence_is_rejected_without_numeric_coercion(field, value):
    with pytest.raises(ValueError):
        run(filings=[disclosure(**{field: value})])


def test_duplicate_ambiguous_vintage_is_rejected_not_arbitrarily_selected():
    second = disclosure(tracing_no='67890', source_sha256=SHA2, amount_rial='600000000')
    with pytest.raises(ValueError, match='Ambiguous'):
        run(filings=[disclosure(), second])


@pytest.mark.parametrize('corruption', [
    {'date': CUTOFF, 'decision_at': '2025-03-21T12:30:00+03:30'},
    {'date': '2024-02-06'},
    {'instrument_id': 'کگل'},
    {'decision_at': '2024-02-05T12:30:00'},
])
def test_invalid_decision_or_holdout_is_rejected(corruption):
    with pytest.raises(ValueError):
        run(decisions=[dict(decision(), **corruption)])


def test_holdout_disclosures_rejected_even_if_no_decision_uses_them():
    sealed = disclosure(published_at='2025-03-21T11:00:00+03:30',
                        first_usable_at='2025-03-21T11:01:00+03:30')
    with pytest.raises(ValueError, match='holdout'):
        run(filings=[disclosure(), sealed])


def test_no_filing_retains_opportunity_as_unknown():
    row = run(filings=[])[0]
    assert row['status'] == 'no_source_observed' and row['amount_rial'] is None


def test_id_and_date_duplicates_are_rejected():
    with pytest.raises(ValueError, match='Duplicate decision'):
        run(decisions=[decision(), deepcopy(decision())])
    with pytest.raises(ValueError, match='Duplicate source'):
        run(filings=[disclosure(), deepcopy(disclosure())])


def test_unverified_missing_numeric_and_availability_remains_unknown():
    row = disclosure(value_verified=False, historical_availability_verified=False,
                     amount_rial=None, first_usable_at=None)
    result = run(filings=[row])[0]
    assert result['status'] == 'historical_evidence_unverified'
    assert result['amount_rial'] is None and result['source_sha256'] is None


def test_verified_flag_cannot_invent_missing_availability_or_amount():
    with pytest.raises(ValueError, match='Verified availability'):
        run(filings=[disclosure(first_usable_at=None)])
    with pytest.raises(ValueError, match='amount'):
        run(filings=[disclosure(amount_rial=None)])


def test_newer_unverified_revision_does_not_hide_behind_old_verified_revenue():
    newer = disclosure(source_sha256=SHA2, tracing_no='67890', amount_rial=None, value_verified=False,
                       published_at='2024-02-05T10:00:00+03:30',
                       first_usable_at=None, historical_availability_verified=False)
    row = run(filings=[disclosure(), newer])[0]
    assert row['status'] == 'newer_vintage_unverified'
    assert row['amount_rial'] is None


@pytest.mark.parametrize('field,value', [
    ('report_kind', 'annual_balance_sheet'), ('period_months', 12),
    ('period_months', True), ('is_consolidated', True), ('currency', 'IRT'),
])
def test_other_report_types_units_and_consolidation_cannot_be_mislabelled_as_monthly_revenue(field, value):
    with pytest.raises(ValueError, match='standalone monthly'):
        run(filings=[disclosure(**{field: value})])
