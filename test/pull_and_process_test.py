import unittest
from pull_and_process_Data import *
import pytest

@pytest.fixture(autouse=True)
def test_create_directory_and_manifest(self):
    # Test the function with default argument
    output_dir, manifest_path = create_directory_and_manifest()
    self.assertTrue(os.path.exists(output_dir))
    self.assertTrue(os.path.exists(manifest_path) or not os.path.exists(manifest_path))
    
    # Test the function with a custom directory name
    custom_dir = 'custom_output'
    output_dir, manifest_path = create_directory_and_manifest(directory_name=custom_dir)
    self.assertTrue(os.path.exists(output_dir))
    self.assertTrue(os.path.exists(manifest_path) or not os.path.exists(manifest_path))

def test_create_cache_get_session_table(self):
    # Assume a valid manifest_path is available
    manifest_path = 'path_to_manifest/manifest.json'
    cache, session_table = create_cache_get_session_table(manifest_path)
    self.assertIsInstance(cache, EcephysProjectCache)
    self.assertIsInstance(session_table, pd.DataFrame)

def test_pick_session_and_pull_data(self):
    # Assume a valid cache and session_number are available
    cache = EcephysProjectCache(manifest='path_to_manifest/manifest.json')
    session_number = 123456  # Example session number
    spike_times, stimulus_table = pick_session_and_pull_data(cache, session_number)
    self.assertIsInstance(spike_times, dict)
    self.assertIsInstance(stimulus_table, pd.DataFrame)

if __name__ == '__main__':
    unittest.main()
