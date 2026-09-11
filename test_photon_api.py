import unittest
from unittest.mock import Mock, patch

import photon_api
from knowledge_relations import location_hierarchy as hierarchy
from knowledge_settings import LOCATION_HIERARCHY_MODULE_VERSION


class PhotonHierarchyTests(unittest.TestCase):
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
        self.assertEqual(tx.run.call_args.kwargs["module_version"], "location-hierarchy-v2-photon")
