import unittest
from unittest.mock import Mock, patch

import knowledge_relations.location_hierarchy as subject


class LocationHierarchyTests(unittest.TestCase):
    def test_content_edge_requires_extracted_locations_and_verbatim_evidence(self):
        locations = [
            {"entity_id": "e1", "node_id": "1", "name": "Thanh Xuân"},
            {"entity_id": "e2", "node_id": "2", "name": "Hà Nội"},
        ]
        edges = subject.normalize_content_edges(
            "Sự kiện tại Thanh Xuân, Hà Nội.",
            locations,
            {"relations": [
                {"source_entity_id": "e1", "target_entity_id": "e2", "evidence_text": "Thanh Xuân, Hà Nội"},
                {"source_entity_id": "e1", "target_entity_id": "missing", "evidence_text": "Thanh Xuân, Hà Nội"},
                {"source_entity_id": "e2", "target_entity_id": "e1", "evidence_text": "không có trong bài"},
            ]},
        )
        self.assertEqual(len(edges), 1)
        self.assertEqual(edges[0]["source"], "CONTENT")

    def test_addressdetails_builds_chain_and_ignores_poi_display_name(self):
        result = subject.administrative_chain("Nam Từ Liêm", [{
            "osm_id": 123,
            "osm_type": "relation",
            "display_name": "Vietnam National Convention Center",
            "address": {"city_district": "Nam Từ Liêm", "state": "Hà Nội", "country": "Việt Nam"},
        }])
        self.assertEqual([item["name"] for item in result["chain"]], ["Nam Từ Liêm", "Hà Nội", "Việt Nam"])
        self.assertNotIn("Vietnam National Convention Center", str(result["chain"]))

    def test_state_is_preferred_over_suburb_as_parent(self):
        result = subject.administrative_chain("Thái Bình Ward", [{
            "address": {
                "suburb": "Thái Bình Ward",
                "town": "Quang Lịch Commune",
                "state": "Hưng Yên Province",
                "ISO3166-2-lvl4": "VN-66",
                "country": "Vietnam",
            },
        }])

        self.assertEqual(
            [item["name"] for item in result["chain"]],
            ["Thái Bình Ward", "Hưng Yên", "Vietnam"],
        )
        self.assertNotIn("Quang Lịch Commune", str(result["chain"]))

    def test_suburb_is_used_when_state_is_missing(self):
        result = subject.administrative_chain("Thôn Đông", [{
            "address": {
                "village": "Thôn Đông",
                "suburb": "Phường Trung Tâm",
                "country": "Việt Nam",
            },
        }])

        self.assertEqual(
            [item["name"] for item in result["chain"]],
            ["Thôn Đông", "Phường Trung Tâm", "Việt Nam"],
        )

    def test_addressdetails_removes_province_and_country_suffixes(self):
        result = subject.administrative_chain("Phú Quốc", [{
            "address": {
                "district": "Phú Quốc",
                "state": "An Giang Province",
                "country": "Vietnam Country",
            },
        }])

        self.assertEqual(
            [item["name"] for item in result["chain"]],
            ["Phú Quốc", "An Giang", "Vietnam"],
        )

    def test_admin_suffix_cleaner_only_removes_trailing_words(self):
        self.assertEqual(subject.clean_osm_admin_name("An Giang Province"), "An Giang")
        self.assertEqual(subject.clean_osm_admin_name("Vietnam Country"), "Vietnam")
        self.assertEqual(subject.clean_osm_admin_name("Province Road"), "Province Road")

    def test_distinct_osm_hierarchies_are_ambiguous(self):
        results = [
            {"address": {"district": "Springfield", "state": "A", "country": "X"}},
            {"address": {"district": "Springfield", "state": "B", "country": "X"}},
        ]
        self.assertIsNone(subject.administrative_chain("Springfield", results))

    def test_multiple_osm_results_are_ambiguous_even_with_same_hierarchy(self):
        results = [
            {
                "osm_id": 1,
                "address": {"suburb": "Hòa Lạc", "city": "Hà Nội", "country": "Việt Nam"},
            },
            {
                "osm_id": 2,
                "address": {"suburb": "Hòa Lạc", "city": "Hà Nội", "country": "Việt Nam"},
            },
        ]

        self.assertIsNone(subject.administrative_chain("Hòa Lạc", results))

    def test_multilingual_state_skips_its_own_english_name(self):
        result = subject.administrative_chain("Tứ Xuyên", [{
            "name": "Sichuan", "namedetails": {"name:vi": "Tứ Xuyên"},
            "address": {"state": "Sichuan", "country": "China"},
        }])
        self.assertEqual([item["name"] for item in result["chain"]], ["Tứ Xuyên", "China"])

    def test_hotel_uses_city_instead_of_neighbourhood(self):
        result = subject.administrative_chain("Rosewood Hồng Kông", [{
            "name": "Rosewood Hong Kong",
            "address": {"tourism": "Rosewood Hong Kong", "suburb": "Tsim Sha Tsui",
                        "region": "Kowloon", "city": "Hong Kong", "country": "China"},
        }])
        self.assertEqual([item["name"] for item in result["chain"]],
                         ["Rosewood Hồng Kông", "Hong Kong", "China"])

    def test_address_match_does_not_identify_a_different_object(self):
        self.assertIsNone(subject.administrative_chain("Hà Nội", [{
            "name": "Hotel Example", "address": {"city": "Hà Nội", "country": "Vietnam"},
        }]))

    def test_brand_and_substring_are_not_object_aliases(self):
        for details in ({"brand": "Rosewood"}, {"name:vi": "Khách sạn Rosewood"}):
            self.assertIsNone(subject.administrative_chain("Rosewood", [{
                "name": "Rosewood Hong Kong", "namedetails": details,
                "address": {"city": "Hong Kong", "country": "China"},
            }]))

    def test_own_alias_without_parent_cannot_build_chain(self):
        self.assertIsNone(subject.administrative_chain("Tứ Xuyên", [{
            "name": "Sichuan", "namedetails": {"name:vi": "Tứ Xuyên"},
            "address": {"state": "Sichuan"},
        }]))

    def test_missing_addressdetails_is_skipped(self):
        self.assertIsNone(subject.administrative_chain("Hà Nội", [{"display_name": "Hà Nội"}]))

    def test_name_variants_resolve_prefixed_city(self):
        variants = subject._search_variants("Hà Nội")
        self.assertIn("ha noi", variants)
        self.assertIn("tp ha noi", variants)

    @patch.object(subject.time, "sleep")
    def test_geocoder_wraps_http_and_json_failures(self, _sleep):
        response = Mock()
        response.raise_for_status.side_effect = RuntimeError("timeout")
        with self.assertRaises(subject.OSMEnrichmentError):
            subject.geocode_location("Nam Từ Liêm", request_get=Mock(return_value=response))

    def test_geocoder_matches_full_name_without_accents(self):
        exact = {"name": "Thái Bình", "address": {"city": "Thái Bình"}}
        response = Mock()
        response.json.return_value = [
            {"name": "Thái Bình Commune", "address": {"town": "Thái Bình Commune"}},
            {"name": "Thái Bình Ward", "address": {"city": "Thái Bình"}},
            exact,
            {"address": {"city": "Thái Bình"}},
        ]
        self.assertEqual(
            subject.geocode_location("  THAI BINH  ", request_get=Mock(return_value=response)),
            [exact],
        )

    def test_geocoder_returns_empty_when_no_full_name_matches(self):
        response = Mock()
        response.json.return_value = [
            {"name": "Bình Minh Commune", "address": {"town": "Xã Bình Minh"}},
        ]
        self.assertEqual(
            subject.geocode_location("xa binh minh", request_get=Mock(return_value=response)),
            [],
        )

    def test_geocoder_empty_query_does_not_request_api(self):
        request_get = Mock()
        self.assertEqual(subject.geocode_location("   ", request_get=request_get), [])
        request_get.assert_not_called()

    @patch.object(subject.time, "sleep")
    def test_geocoder_rejects_payload_without_addressdetails(self, _sleep):
        response = Mock()
        response.json.return_value = [{"display_name": "POI only"}]
        with self.assertRaises(subject.OSMEnrichmentError):
            subject.geocode_location("Nam Từ Liêm", request_get=Mock(return_value=response))

    def test_osm_failure_marks_only_hierarchy_failed(self):
        session = Mock()
        locations_result = [{
            "node_id": "node-1", "name": "Nam Từ Liêm",
            "normalized_name": "nam từ liêm", "aliases": ["nam từ liêm"],
            "search_name": "nam tu liem", "osm_id": None, "osm_type": None,
        }]
        parent_result = Mock()
        parent_result.single.return_value = {"has_parent": False}
        status_result = Mock()
        session.run.side_effect = [locations_result, parent_result, status_result]
        knowledge = {"entities": [{
            "local_id": "e1", "name": "Nam Từ Liêm",
            "canonical_name": "Nam Từ Liêm", "type": "LOCATION",
        }]}

        result = subject.enrich_location_hierarchy(
            session, "facebook", "1", "Nam Từ Liêm", knowledge,
            call_model=Mock(),
            geocode_fn=Mock(side_effect=subject.OSMEnrichmentError("timeout")),
        )

        self.assertEqual(result["errors"], 1)
        self.assertEqual(session.run.call_args.kwargs["status"], "FAILED")
        session.execute_write.assert_not_called()


class ContextGeocodingTests(unittest.TestCase):
    def request(self, *payloads):
        responses = []
        for payload in payloads:
            response = Mock()
            response.json.return_value = payload
            responses.append(response)
        return Mock(side_effect=responses)

    def test_scoring_and_unicode(self):
        for name, expected in [("Hải Phòng", 1.0), ("Hải Phong", 0.8),
                               ("Cảng Hải Phòng", 0.6), ("Cang Hai Phong", 0.4), ("Other", 0.0)]:
            with self.subTest(name=name):
                self.assertEqual(subject.match_score("hải phòng", {"name": name}), expected)
        self.assertEqual(subject.normalize_vi(" ĐẮK ", strip_accents=True), "dak")
        self.assertEqual(subject.normalize_vi("ắ"), subject.normalize_vi("a\u0306\u0301"))
        self.assertEqual(subject.match_score(" ", {"name": "anything"}), 0)
        self.assertEqual(subject.match_score("Hà Nội", {"address": {"city": "Hà Nội"},
                                                          "namedetails": {"bad": None}}), 1)

    def test_hai_phong_beats_haifeng(self):
        vn = {"name": "Hải Phòng", "address": {"city": "Hải Phòng", "country": "Vietnam"}}
        cn = {"name": "Haifeng County", "address": {"country": "China"},
              "namedetails": {"name:vi": "Hải Phong"}}
        self.assertEqual(subject.match_score("hải phòng", cn), 0.8)
        self.assertEqual(subject.geocode_with_hints("hải phòng", request_get=self.request([cn, vn])), [vn])

    def test_structured_town_real_response_fixture(self):
        # Observed Nominatim town response; this does not establish commune boundary support.
        town = {"osm_type": "node", "osm_id": 13778682314, "type": "town", "name": "Hoà Lạc",
                "address": {"town": "Hoà Lạc", "city": "Hà Nội", "country": "Vietnam"},
                "namedetails": {"name": "Hoà Lạc"}}
        park = {"osm_type": "relation", "osm_id": 19849694, "type": "industrial",
                "name": "Hoa Lac Hi-Tech Park",
                "address": {"industrial": "Hoa Lac Hi-Tech Park", "city_district": "Hoa Lac Commune",
                            "city": "Hà Nội", "country": "Vietnam"},
                "namedetails": {"name": "Khu Công nghệ cao Hòa Lạc", "name:vi": "Khu Công nghệ cao Hòa Lạc"}}
        request = self.request([park, town])
        self.assertEqual(subject.match_score("hòa lạc", town), 0.8)
        self.assertEqual(subject.match_score("hòa lạc", park), 0.6)
        self.assertEqual(subject.geocode_with_hints("hòa lạc", ["hà nội"], request), [town])
        params = request.call_args.kwargs["params"]
        self.assertEqual(params["city"], "hòa lạc")
        self.assertEqual(params["state"], "hà nội")
        self.assertNotIn("q", params)
        self.assertEqual(params["namedetails"], 1)

    def test_ties_preserve_positive_candidates_and_order(self):
        rows = [{"name": "Hòa Lạc", "address": {"state": state}} for state in ("A", "B")]
        rows.append({"name": "Khu Hòa Lạc", "address": {"state": "C"}})
        result = subject.geocode_with_hints("hòa lạc", request_get=self.request(rows))
        self.assertEqual(result, rows)
        self.assertIsNone(subject.administrative_chain("hòa lạc", result))

    def test_unaccented_name_builds_chain(self):
        row = {"name": "Hà Nội", "address": {"city": "Hà Nội", "country": "Vietnam"}}
        result = subject.geocode_with_hints("ha noi", request_get=self.request([row]))
        self.assertEqual(subject.match_score("ha noi", row), 0.8)
        self.assertEqual(subject.administrative_chain("ha noi", result)["chain"][-1]["name"], "Vietnam")

    def test_structured_tie_then_city_and_raw_fallback(self):
        rows = [{"name": "Hòa Lạc", "address": {"city": "Hà Nội"}, "osm_id": i} for i in (1, 2)]
        request = self.request(rows, rows, rows)
        self.assertEqual(subject.geocode_with_hints("hòa lạc", ["hà nội"], request), rows)
        params = [call.kwargs["params"] for call in request.call_args_list]
        self.assertEqual(params[1]["street"], "hòa lạc")
        self.assertEqual(params[1]["city"], "hà nội")
        self.assertEqual(params[2]["q"], "hòa lạc")

    def test_wrong_hint_is_rejected_before_ranking(self):
        wrong = {"name": "Hòa Lạc", "address": {"state": "Hà Nội mở rộng"}}
        right = {"name": "Hoà Lạc", "address": {"city": "Ha Noi Province"}}
        request = self.request([wrong], [wrong, right])
        self.assertEqual(subject.geocode_with_hints("hòa lạc", ["hà nội"], request), [right])
        self.assertEqual(request.call_count, 2)

    def test_empty_and_duplicate_hints(self):
        request = self.request([], [], [])
        self.assertEqual(subject.geocode_with_hints("hòa lạc", ["", "hòa lạc", "hà nội", "HA NOI"], request), [])
        self.assertEqual(request.call_count, 3)
        empty_request = Mock()
        self.assertEqual(subject.geocode_with_hints(" ", ["Hà Nội"], empty_request), [])
        empty_request.assert_not_called()

    def test_raw_preserves_address_and_namedetails_matches(self):
        rows = [{"name": "Other", "address": {"city": "Hà Nội"}},
                {"address": {"country": "Vietnam"}, "namedetails": {"name:vi": "Hà Nội"}}]
        self.assertEqual(subject.geocode_location("Hà Nội", self.request(rows)), [])
        self.assertEqual(subject.geocode_location("Hà Nội", self.request(rows), raw=True), rows)

    def test_raw_fallback_resolves_binh_minh_using_dong_nai_hint(self):
        rows = [
            {"name": "Xã Bình Minh", "osm_id": osm_id,
             "address": {"county": "Xã Bình Minh", "state": state, "country": "Vietnam"}}
            for osm_id, state in [(13474383, "Đồng Nai"), (19369667, "Quảng Ngãi Province")]
        ]
        request = self.request([], [], [], [], rows)
        result = subject.geocode_with_hints("xã Bình Minh", ["ấp Bùi Chu", "dong nai"], request)
        self.assertEqual(result, [rows[0]])
        self.assertEqual(request.call_count, 5)
        self.assertEqual(request.call_args.kwargs["params"]["q"], "xã Bình Minh")
        self.assertEqual(
            [item["name"] for item in subject.administrative_chain("xã Bình Minh", result)["chain"]],
            ["xã Bình Minh", "Đồng Nai", "Vietnam"],
        )

    def test_raw_hint_matches_city_and_respects_hint_order(self):
        rows = [{"name": "Bình Minh", "address": {"city": city}}
                for city in ["Hà Nội", "Đồng Nai"]]
        request = self.request([], [], [], [], rows)
        self.assertEqual(subject.geocode_with_hints("Bình Minh", ["dong nai", "Hà Nội"], request), [rows[1]])

    def test_raw_hint_tie_does_not_guess(self):
        rows = [{"name": "Bình Minh", "osm_id": i, "address": {"state": "Đồng Nai Province"}}
                for i in (1, 2)]
        request = self.request([], [], rows)
        self.assertEqual(subject.geocode_with_hints("Bình Minh", ["Đồng Nai"], request), rows)

    def test_raw_hint_requires_full_address_match(self):
        rows = [{"name": "Bình Minh", "address": {"state": state}}
                for state in ["Đồng Nai mở rộng", "Quảng Ngãi"]]
        request = self.request([], [], rows)
        self.assertEqual(subject.geocode_with_hints("Bình Minh", ["Đồng Nai"], request), rows)

    @patch.object(subject, "extract_content_edges", return_value=[])
    @patch.object(subject, "load_post_locations")
    def test_ambiguity_marks_entity_review(self, load, extract):
        load.return_value = [{"node_id": "1", "name": "Hòa Lạc", "normalized_name": "hòa lạc"}]
        session = Mock()
        session.run.return_value.single.return_value = {"has_parent": False}
        rows = [{"name": "Hòa Lạc", "address": {"state": state}} for state in ("A", "B")]
        request = self.request(rows)
        def geocode(query, hints=None):
            return subject.geocode_with_hints(query, hints, request)
        result = subject.enrich_location_hierarchy(session, "facebook", "1", "Hòa Lạc",
            {"entities": [{"type": "LOCATION", "name": "Hòa Lạc"}]}, geocode_fn=geocode)
        self.assertEqual(result["skipped"], 1)
        self.assertEqual(result["errors"], 0)
        session.execute_write.assert_not_called()
        review = [call for call in session.run.call_args_list if "SET location.needs_review = true" in call.args[0]]
        self.assertEqual(len(review), 1)
        self.assertEqual(review[0].kwargs["node_id"], "1")

    @patch.object(subject, "extract_content_edges")
    @patch.object(subject, "load_post_locations")
    def test_hints_prioritize_only_persisted_edges(self, load, extract):
        names = ["Target", "Sibling", "Child", "Parent", "Rejected"]
        load.return_value = [{"node_id": n, "name": n, "normalized_name": n.lower()} for n in names]
        saved = {"source_node_id": "Child", "target_node_id": "Parent"}
        rejected = {"source_node_id": "Sibling", "target_node_id": "Rejected"}
        extract.return_value = [saved, rejected]
        session = Mock()
        session.execute_write.return_value = {"created": 1, "skipped": 1, "persisted_edges": [saved]}
        session.run.return_value.single.return_value = {"has_parent": False}
        geocode = Mock(return_value=[])
        subject.enrich_location_hierarchy(session, "facebook", "1", "",
            {"entities": [{"type": "LOCATION", "name": n} for n in names]}, geocode_fn=geocode)
        self.assertEqual(geocode.call_args_list[0].kwargs["hints"], ["Parent", "Child", "Sibling", "Rejected"])

    def test_persistence_reports_existing_edges_but_not_rejected_edges(self):
        tx = Mock()
        tx.run.return_value.single.side_effect = [{"created": True}, None, {"created": False}]
        edges = [{"source_node_id": str(i), "target_node_id": "parent"} for i in range(3)]
        result = subject._persist_edges_tx(tx, edges)
        self.assertEqual(result["persisted_edges"], [edges[0], edges[2]])
        self.assertEqual((result["created"], result["skipped"]), (1, 1))


if __name__ == "__main__":
    unittest.main()
