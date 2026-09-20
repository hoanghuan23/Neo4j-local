import unittest
from datetime import date, datetime, timezone
from unittest.mock import Mock

from scripts.backfill_event_occurrence_dates import backfill_occurrence_dates
from event_time import normalize_occurrence_date


class EventTimeNormalizationTests(unittest.TestCase):
    def normalize(self, expression, posted_at=datetime(2026, 9, 18, 10, 0)):
        return normalize_occurrence_date(expression, posted_at)

    def test_relative_vietnamese_dates(self):
        cases = {
            "hôm nay": date(2026, 9, 18),
            "hôm qua": date(2026, 9, 17),
            "hôm kia": date(2026, 9, 16),
            "3 ngày trước": date(2026, 9, 15),
            "ngày mai": date(2026, 9, 19),
        }
        for expression, expected in cases.items():
            with self.subTest(expression=expression):
                result = self.normalize(expression)
                self.assertEqual(result["occurrence_date"], expected)
                self.assertEqual(result["occurrence_date_source"], "RELATIVE")

    def test_explicit_date_has_priority_over_relative_word(self):
        result = self.normalize("Hôm qua (16/9), sự kiện xảy ra")
        self.assertEqual(result["occurrence_date"], date(2026, 9, 16))
        self.assertEqual(result["occurrence_date_source"], "EXPLICIT")

    def test_vietnamese_written_date(self):
        result = self.normalize("Ngày 19 tháng 9 năm 1954")
        self.assertEqual(result["occurrence_date"], date(1954, 9, 19))
        self.assertEqual(result["occurrence_date_source"], "EXPLICIT")

    def test_missing_year_uses_posted_year(self):
        result = self.normalize("Ngày 18/9")
        self.assertEqual(result["occurrence_date"], date(2026, 9, 18))
        self.assertEqual(result["occurrence_date_source"], "EXPLICIT")

    def test_no_expression_does_not_persist_posted_date(self):
        result = normalize_occurrence_date(None, datetime(2026, 9, 18, 10, 0))
        self.assertIsNone(result["occurrence_date"])
        self.assertIsNone(result["occurrence_date_source"])

    def test_unrelated_event_evidence_does_not_persist_posted_date(self):
        result = normalize_occurrence_date(
            None,
            datetime(2026, 9, 20, 10, 0),
            evidence_text=(
                "Công an tiếp tục điều tra vụ tai nạn nghiêm trọng "
                "xảy ra tại Hà Nội."
            ),
        )
        self.assertIsNone(result["occurrence_date"])
        self.assertIsNone(result["occurrence_date_source"])

    def test_relative_keyword_can_come_from_event_evidence(self):
        result = normalize_occurrence_date(
            None,
            datetime(2026, 9, 18, 10, 0),
            evidence_text="Hôm qua tại thành phố B xảy ra sự kiện A",
        )
        self.assertEqual(result["occurrence_date"], date(2026, 9, 17))
        self.assertEqual(result["occurrence_date_source"], "RELATIVE")

    def test_utc_timestamp_is_converted_to_vietnam_date(self):
        result = normalize_occurrence_date(
            "hôm nay", datetime(2026, 9, 18, 18, 0, tzinfo=timezone.utc)
        )
        self.assertEqual(result["occurrence_date"], date(2026, 9, 19))

    def test_backfill_is_dry_run_by_default(self):
        session = Mock()
        session.run.return_value = [{
            "mention_key": "m1",
            "time_expression": "hôm qua",
            "evidence_text": "Hôm qua xảy ra sự kiện A",
            "posted_at": datetime(2026, 9, 18, 10, 0),
        }]

        result = backfill_occurrence_dates(session)

        self.assertEqual(result["selected"], 1)
        self.assertEqual(result["updated"], 0)
        self.assertEqual(result["mentions"][0]["occurrence_date"], "2026-09-17")
        self.assertEqual(session.run.call_count, 1)

    def test_backfill_apply_clears_legacy_posted_at_fallback(self):
        session = Mock()
        write_result = Mock()
        session.run.side_effect = [[{
            "mention_key": "m1",
            "time_expression": None,
            "evidence_text": "Công an tiếp tục điều tra vụ tai nạn tại Hà Nội",
            "posted_at": datetime(2026, 9, 20, 10, 0),
        }], write_result]

        result = backfill_occurrence_dates(session, apply=True)

        self.assertEqual(result["sources"]["UNRESOLVED"], 1)
        rows = session.run.call_args_list[1].kwargs["rows"]
        self.assertIsNone(rows[0]["occurrence_date"])
        self.assertIsNone(rows[0]["occurrence_date_source"])


if __name__ == "__main__":
    unittest.main()
