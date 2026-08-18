import unittest
from unittest.mock import Mock, call

import import_facebook_from_postgreSQL as subject


class FacebookPostSyncTests(unittest.TestCase):
    def test_updates_recent_existing_posts_when_there_are_no_new_posts(self):
        session = Mock()
        session.run.return_value = [
            {"platform_id": "recent"},
            {"platform_id": "old"},
        ]
        posts = [
            {
                "facebook_post_id": "recent",
                "metric_tier": "hot",
                "should_refresh_metric_tier": True,
            },
            {
                "facebook_post_id": "old",
                "metric_tier": "cold",
                "should_refresh_metric_tier": False,
            },
        ]

        stats = subject.sync_posts(session, posts)

        self.assertEqual(stats["new_posts"], 0)
        self.assertEqual(stats["updated_metric_tiers"], 1)
        self.assertEqual(
            session.execute_write.call_args_list,
            [call(subject.update_post_metric_tier, posts[0])],
        )

    def test_imports_new_posts_and_only_refreshes_existing_recent_posts(self):
        session = Mock()
        session.run.return_value = [{"platform_id": "existing"}]
        new_post = {
            "facebook_post_id": "new",
            "should_refresh_metric_tier": True,
        }
        existing_post = {
            "facebook_post_id": "existing",
            "metric_tier": "warm",
            "should_refresh_metric_tier": True,
        }

        stats = subject.sync_posts(session, [new_post, existing_post])

        self.assertEqual(stats["new_posts"], 1)
        self.assertEqual(stats["updated_metric_tiers"], 1)
        self.assertEqual(
            session.execute_write.call_args_list,
            [
                call(subject.import_post, new_post),
                call(subject.update_post_metric_tier, existing_post),
            ],
        )


if __name__ == "__main__":
    unittest.main()
