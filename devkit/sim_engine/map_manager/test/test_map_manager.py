import unittest
from unittest.mock import Mock, patch
import sys

sys.path.append("/home/czf/project_czf/20240901-MineSim/MineSim-Dynamic-Dev-czf/MineSim-Dynamic-Dev")

from devkit.sim_engine.map_manager.map_manager import MapManager


class TestMapManager(unittest.TestCase):
    """
    MapManager test suite.
    """

    @patch("devkit.sim_engine.map_manager.abstract_map_factory import AbstractMapFactory")
    def setUp(self, mock_map_factory: Mock) -> None:
        """
        Initializes the map manager.
        """
        self.map_manager = MapManager(mock_map_factory)
        a=1

    @patch("devkit.sim_engine.map_manager.abstract_map_factory import AbstractMapFactory")
    def test_initialization(self, mock_map_factory: Mock) -> None:
        """Tests that objects are initialized correctly."""
        map_manager = MapManager(mock_map_factory)

        self.assertEqual(mock_map_factory, map_manager.map_factory)
        self.assertEqual({}, map_manager.maps)

    def test_get_map(self) -> None:
        """Tests that maps are retrieved from cache, if not present created and added to it."""
        map_name = "map_name"
        self.map_manager.map_factory.build_map_from_name.return_value = "built_map"

        _map = self.map_manager.get_map(map_name)
        # We expect the map to be built and to be in the dict
        self.map_manager.map_factory.build_map_from_name.assert_called_once_with(map_name)
        self.assertTrue(map_name in self.map_manager.maps)
        self.assertEqual("built_map", _map)
        # If we call the get map again, we expect it to be cached
        _ = self.map_manager.get_map(map_name)
        self.map_manager.map_factory.build_map_from_name.assert_called_once_with(map_name)
        a = 1


if __name__ == "__main__":
    unittest.main()
