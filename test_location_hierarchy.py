import unittest
from unittest.mock import Mock, patch

import knowledge_relations.location_hierarchy as subject


class LocationHierarchyTests(unittest.TestCase):
    def test_preview_uses_explicit_province_to_resolve_lao_cai(self):
        import preview_location_content as preview

        content = "Trường tại xã Mường Khương, tỉnh Lào Cai."
        knowledge = {"entities": [
            {"local_id": "e1", "type": "LOCATION", "name": "Mường Khương"},
            {"local_id": "e2", "type": "LOCATION", "name": "Lào Cai"},
        ]}
        province = {"name": "Lào Cai Province", "address": {
            "state": "Lào Cai Province", "country": "Vietnam"},
            "namedetails": {"name:vi": "Tỉnh Lào Cai"}}
        city = {"name": "Lao Cai", "address": {
            "city": "Lao Cai", "state": "Lào Cai Province", "country": "Vietnam"},
            "namedetails": {"name:vi": "Thành phố Lào Cai"}}
        ward = {"name": "Lao Cai Ward", "address": {
            "suburb": "Lao Cai Ward", "city": "Lao Cai",
            "state": "Lào Cai Province", "country": "Vietnam"},
            "namedetails": {"name:vi": "Phường Lào Cai"}}
        rows = [province, city, ward]
        geocode = Mock(return_value=rows)
        with patch.object(preview, "extract_knowledge", return_value=knowledge), patch.object(
            preview, "extract_content_edges", return_value=[{
                "source_node_id": "preview-1", "target_node_id": "preview-2",
                "evidence_text": "xã Mường Khương, tỉnh Lào Cai",
            }]
        ):
            result = preview.preview_content(content, Mock(), geocode)
        resolved = result["locations"][1]
        self.assertEqual(resolved["status"], "RESOLVED")
        self.assertEqual(resolved["part_of_chain"], ["Lào Cai", "Vietnam"])
        self.assertFalse(resolved["needs_review"])
        self.assertEqual(len(resolved["candidates"]), 1)

        location = {"name": "Lào Cai", "node_id": "1"}
        for text in ("Lào Cai", "tỉnh Lào Cai và phường Lào Cai"):
            self.assertEqual(subject.resolve_content_location(
                location, [location], [], [], geocode, content=text), rows)
        # Parent address fields must not identify the city as a province.
        self.assertEqual(subject.resolve_content_location(
            location, [location], [], [], Mock(return_value=[city]), content=content), [])

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

    def test_explicit_street_address_content_parent(self):
        content = "Cận cảnh tại số 96 phố Cầu Đất - Hải Phòng."
        locations = [
            {"node_id": "1", "name": "96 phố Cầu Đất", "normalized_name": "96 phố cầu đất"},
            {"node_id": "2", "name": "Hải Phòng", "normalized_name": "hải phòng"},
        ]
        knowledge = {"entities": [
            {"local_id": "e1", "type": "LOCATION", "name": "96 phố Cầu Đất"},
            {"local_id": "e2", "type": "LOCATION", "name": "Hải Phòng"},
        ]}
        model = Mock(return_value={"relations": [{
            "source_entity_id": "e1", "target_entity_id": "e2",
            "evidence_text": "số 96 phố Cầu Đất - Hải Phòng",
        }]})
        edges = subject.extract_content_edges(content, knowledge, locations, call_model=model)
        self.assertEqual(len(edges), 1)
        self.assertEqual((edges[0]["source_node_id"], edges[0]["target_node_id"]), ("1", "2"))
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
            geocode_fn=Mock(side_effect=RuntimeError("timeout")),
        )

        self.assertEqual(result["errors"], 1)
        self.assertEqual(session.run.call_args.kwargs["status"], "FAILED")
        session.execute_write.assert_not_called()


class ContextGeocodingTests(unittest.TestCase):
    def test_content_descendants_resolve_dong_anh_and_exclude_station(self):
        locations = [{"node_id": name, "name": name} for name in ("Đông Anh", "Ủy Nỗ", "Phúc Lộc")]
        edges = [{"source_node_id": "Ủy Nỗ", "target_node_id": "Đông Anh"},
                 {"source_node_id": "Phúc Lộc", "target_node_id": "Ủy Nỗ"}]
        station = {"name": "Dong Anh", "namedetails": {"name:vi": "Đông Anh"},
                   "address": {"railway": "Dong Anh", "city": "Hà Nội", "country": "Vietnam"}}
        town = {"name": "Đông Anh", "address": {"town": "Đông Anh", "city": "Hà Nội", "country": "Vietnam"}}
        village = {"name": "Đông Anh", "address": {"village": "Đông Anh", "state": "Lâm Đồng Province", "country": "Vietnam"}}
        candidates = [station, town, village]
        child = {"name": "Ủy Nỗ", "address": {"city": "Hà Nội", "country": "Vietnam"}}
        geocode = Mock(side_effect=[candidates, [child], []])
        result = subject.resolve_content_location(locations[0], locations, edges, ["Ủy Nỗ", "Phúc Lộc"], geocode)
        self.assertEqual(result, [town])
        self.assertEqual([item["name"] for item in subject.administrative_chain("Đông Anh", result)["chain"]],
                         ["Đông Anh", "Hà Nội", "Vietnam"])

        for child_rows in ([], [dict(child, name="Khách sạn Ủy Nỗ")],
                           [child, {"name": "Ủy Nỗ", "address": {"state": "Lâm Đồng", "country": "Vietnam"}}]):
            with self.subTest(child_rows=child_rows):
                geocode = Mock(side_effect=[candidates, child_rows, []])
                self.assertEqual(subject.resolve_content_location(locations[0], locations, edges, [], geocode), candidates)

        conflicting = {"name": "Phúc Lộc", "address": {"state": "Lâm Đồng", "country": "Vietnam"}}
        geocode = Mock(side_effect=[candidates, [child], [conflicting]])
        self.assertEqual(subject.resolve_content_location(locations[0], locations, edges, [], geocode), candidates)
        geocode = Mock(return_value=candidates)
        self.assertEqual(subject.resolve_content_location(locations[0], locations, [], ["Ủy Nỗ"], geocode), candidates)
        geocode.assert_called_once()

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

    @patch.object(subject, "extract_content_edges", return_value=[])
    @patch.object(subject, "load_post_locations")
    def test_ambiguity_marks_entity_review(self, load, extract):
        load.return_value = [{"node_id": "1", "name": "Hòa Lạc", "normalized_name": "hòa lạc"}]
        session = Mock()
        session.run.return_value.single.return_value = {"has_parent": False}
        rows = [{"name": "Hòa Lạc", "address": {"state": state}} for state in ("A", "B")]
        geocode = Mock(return_value=rows)
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
