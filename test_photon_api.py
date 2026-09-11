import unittest
from unittest.mock import Mock, patch

import photon_api
from knowledge_relations import location_hierarchy as hierarchy
from knowledge_settings import LOCATION_HIERARCHY_MODULE_VERSION


class PhotonHierarchyTests(unittest.TestCase):
    @patch.object(photon_api, "geocode_with_hints", return_value=[])
    def test_content_resolution_defaults_to_photon(self, geocode):
        location = {"node_id": "1", "name": "Tràng Tiền"}
        self.assertEqual(hierarchy.resolve_content_location(
            location, [location], [], ["Hà Nội"]), [])
        geocode.assert_called_once_with("Tràng Tiền", hints=["Hà Nội"])

    def test_empty_query_does_not_request(self):
        request = Mock()
        self.assertEqual(photon_api.geocode_with_hints(" ", request_get=request), [])
        request.assert_not_called()

    def test_http_errors_are_propagated(self):
        response = Mock()
        response.raise_for_status.side_effect = RuntimeError("timeout")
        with self.assertRaisesRegex(RuntimeError, "timeout"):
            photon_api.geocode_with_hints("Hà Nội", request_get=Mock(return_value=response))

    def test_invalid_payload_is_rejected(self):
        for payload in ([], {}, {"features": None}, {"features": [{}]}):
            with self.subTest(payload=payload):
                response = Mock()
                response.json.return_value = payload
                with self.assertRaises(ValueError):
                    photon_api.geocode_with_hints("Hà Nội", request_get=Mock(return_value=response))

    def test_district_hint_is_verified_and_duplicates_are_skipped(self):
        wrong = Mock()
        wrong.json.return_value = {"features": [{"properties": {
            "name": "Tràng Tiền", "district": "Cửa Nam mở rộng"}}]}
        right = Mock()
        right.json.return_value = {"features": [{"properties": {
            "name": "Tràng Tiền", "district": "Cửa Nam", "city": "Hà Nội"}}]}
        request = Mock(side_effect=[wrong, right])
        rows = photon_api.geocode_with_hints(
            "Tràng Tiền", ["", "Tràng Tiền", "Cửa Nam", "CUA NAM"], request)
        self.assertEqual(rows[0]["address"]["district"], "Cửa Nam")
        self.assertEqual([call.kwargs["params"]["q"] for call in request.call_args_list],
                         ["Tràng Tiền, Cửa Nam", "Tràng Tiền"])
        self.assertTrue(all(call.args[0] == photon_api.PHOTON_URL
                            for call in request.call_args_list))

    def test_limits_results_even_when_server_returns_extra_features(self):
        response = Mock()
        feature = {"properties": {
            "name": "Miami", "city": "Miami", "state": "Florida",
            "country": "United States", "countrycode": "US",
            "osm_type": "R", "osm_id": 123, "osm_key": "place",
        }}
        response.json.return_value = {"features": [feature, feature]}
        request = Mock(return_value=response)
        rows = photon_api.geocode_with_hints("Miami", request_get=request)
        self.assertEqual(len(rows), 1)
        self.assertEqual(request.call_args.kwargs["params"]["limit"], 1)
        self.assertEqual(rows[0]["osm_type"], "relation")
        self.assertEqual(rows[0]["address"]["country_code"], "us")
        self.assertIsNotNone(hierarchy.administrative_chain("Miami", rows))

    def test_empty_response_does_not_invent_candidate(self):
        response = Mock()
        response.json.return_value = {"features": []}
        self.assertEqual(photon_api.geocode_with_hints(
            "Miami", request_get=Mock(return_value=response)), [])

    @patch.object(hierarchy, "extract_content_edges", return_value=[])
    @patch.object(hierarchy, "load_post_locations")
    @patch.object(photon_api, "geocode_with_hints")
    def test_enrichment_defaults_to_photon_and_persists_version(self, geocode, load, extract):
        load.return_value = [{"node_id": "1", "name": "Miami", "normalized_name": "miami"}]
        geocode.return_value = [{"name": "Miami", "address": {
            "city": "Miami", "state": "Florida", "country": "United States"}}]
        session = Mock()
        session.run.return_value.single.return_value = {"has_parent": False}
        session.execute_write.return_value = {
            "parents_created": 2, "parents_reused": 0, "osm_edges": 2, "skipped": 0}
        result = hierarchy.enrich_location_hierarchy(
            session, "facebook", "1", "Miami",
            {"entities": [{"type": "LOCATION", "name": "Miami"}]}, call_model=Mock())
        geocode.assert_called_once_with("Miami", hints=[])
        self.assertEqual(result["osm_edges"], 2)
        self.assertEqual(session.run.call_args.kwargs["version"], LOCATION_HIERARCHY_MODULE_VERSION)
        tx = Mock()
        tx.run.return_value.single.return_value = {"created": True}
        hierarchy._persist_edges_tx(tx, [{
            "source_node_id": "1", "target_node_id": "2", "source": "PHOTON",
            "evidence_text": None, "parent_level": 11, "osm_id": 123, "osm_type": "relation"}])
        self.assertEqual(tx.run.call_args.kwargs["module_version"], LOCATION_HIERARCHY_MODULE_VERSION)

    def test_photon_district_is_kept_in_chain(self):
        row = photon_api._candidate({"properties": {
            "name": "Tràng Tiền Plaza", "district": "Cửa Nam",
            "city": "Hà Nội", "state": "Hà Nội", "country": "Việt Nam",
            "osm_id": 570571856, "osm_type": "W",
        }})
        chain = hierarchy.administrative_chain("Tràng Tiền Plaza", [row])
        self.assertEqual([item["name"] for item in chain["chain"]],
                         ["Tràng Tiền Plaza", "Cửa Nam", "Hà Nội", "Việt Nam"])
        self.assertIn("cua nam", hierarchy._search_variants("Cửa Nam"))
        # A partial name must not silently identify a different location.
        self.assertIsNone(hierarchy.administrative_chain("Tràng Tiền", [row]))

    def test_city_is_not_attached_to_its_district(self):
        chain = hierarchy.administrative_chain("Hà Nội", [{
            "name": "Hà Nội", "address": {"city": "Hà Nội",
                "district": "Cửa Nam", "country": "Việt Nam"}}])
        self.assertEqual([item["name"] for item in chain["chain"]],
                         ["Hà Nội", "Việt Nam"])
