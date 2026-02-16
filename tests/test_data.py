"""
Unit tests for data module: subset/probe indices and loaders.
"""
import shutil
import tempfile
import unittest
from pathlib import Path


class TestDataIndices(unittest.TestCase):
    def setUp(self):
        self.data_dir = tempfile.mkdtemp(prefix="lsi_test_data_")

    def tearDown(self):
        if Path(self.data_dir).exists():
            shutil.rmtree(self.data_dir, ignore_errors=True)

    def test_train_subset_indices_same_seed(self):
        from data.indices import get_train_subset_indices

        idx1 = get_train_subset_indices(
            64, dataset_seed=999, data_dir=self.data_dir, num_train=500
        )
        idx2 = get_train_subset_indices(
            64, dataset_seed=999, data_dir=self.data_dir, num_train=500
        )
        self.assertEqual(len(idx1), 64)
        self.assertEqual(idx1, idx2)
        self.assertNotEqual(sorted(idx1), list(range(64)))

    def test_probe_indices_same_seed(self):
        from data.indices import get_probe_indices

        idx1 = get_probe_indices(
            32, dataset_seed=888, data_dir=self.data_dir, num_test=500
        )
        idx2 = get_probe_indices(
            32, dataset_seed=888, data_dir=self.data_dir, num_test=500
        )
        self.assertEqual(len(idx1), 32)
        self.assertEqual(idx1, idx2)


class TestDataLoaders(unittest.TestCase):
    def setUp(self):
        self.data_dir = tempfile.mkdtemp(prefix="lsi_test_data_")

    def tearDown(self):
        if Path(self.data_dir).exists():
            shutil.rmtree(self.data_dir, ignore_errors=True)

    def test_train_loader_shapes(self):
        from data.cifar import get_train_loader

        loader = get_train_loader(
            n=64,
            batch_size=32,
            dataset_seed=42,
            data_dir=self.data_dir,
            root="./data",
        )
        batch = next(iter(loader))
        x, y = batch
        self.assertEqual(x.shape, (32, 3, 32, 32))
        self.assertEqual(y.shape, (32,))

    def test_probe_loader_shapes(self):
        from data.cifar import get_probe_loader

        loader = get_probe_loader(
            probe_size=32,
            dataset_seed=43,
            data_dir=self.data_dir,
            root="./data",
        )
        batch = next(iter(loader))
        x, y = batch
        self.assertEqual(x.shape[0], 32)
        self.assertEqual(x.shape[1], 3)
        self.assertEqual(x.shape[2], 32)
        self.assertEqual(x.shape[3], 32)
        self.assertEqual(y.shape, (32,))


if __name__ == "__main__":
    unittest.main()
