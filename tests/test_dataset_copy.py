"""Filesystem protection checks; SAM inference is replaced by a fixture."""
import runpy
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

process_dataset = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'scripts/preprocessing.py'))['process_dataset']


class DatasetCopyTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.source = self.root / 'source'
        (self.source / 'train/images').mkdir(parents=True)
        (self.source / 'train/labels').mkdir(parents=True)
        self.image = self.source / 'train/images/egg.jpg'
        self.image.write_bytes(b'image-fixture')
        self.label = self.source / 'train/labels/egg.txt'
        self.label.write_text('0 0.5 0.5 0.2 0.2\n')
        self.checkpoint = self.root / 'sam.pt'
        self.checkpoint.write_bytes(b'checkpoint-fixture')

    def test_output_cannot_be_input(self):
        with self.assertRaises(ValueError):
            process_dataset(self.source, self.source, self.checkpoint)

    def test_output_cannot_be_inside_input(self):
        with self.assertRaises(ValueError):
            process_dataset(self.source, self.source/'converted', self.checkpoint)
        self.assertFalse((self.source/'converted').exists())

    def test_existing_output_is_preserved(self):
        output = self.root/'output'
        output.mkdir()
        sentinel = output/'keep.txt'
        sentinel.write_text('keep')
        with self.assertRaises(ValueError):
            process_dataset(self.source, output, self.checkpoint)
        self.assertEqual(sentinel.read_text(), 'keep')

    def test_missing_checkpoint_creates_no_output(self):
        output = self.root/'output'
        with self.assertRaises(ValueError):
            process_dataset(self.source, output, self.root/'missing.pt')
        self.assertFalse(output.exists())

    def test_conversion_changes_only_the_new_dataset_copy(self):
        output = self.root/'output'
        model = Mock()
        sam = types.SimpleNamespace(sam_model_registry={'vit_b': Mock(return_value=model)}, SamPredictor=Mock())
        torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))

        def fixture_conversion(image, label, predictor):
            Path(label).write_text('0 0.1 0.1 0.2 0.1 0.2 0.2\n')
            return True

        with patch.dict(sys.modules, {'torch': torch, 'segment_anything': sam}), patch.dict(process_dataset.__globals__, {'bbox_to_mask': fixture_conversion}):
            process_dataset(self.source, output, self.checkpoint)
        self.assertEqual(self.label.read_text(), '0 0.5 0.5 0.2 0.2\n')
        self.assertTrue((output/'train/labels/egg.txt').read_text().startswith('0 0.1'))
        self.assertEqual((output/'train/images/egg.jpg').read_bytes(), self.image.read_bytes())


if __name__ == '__main__':
    unittest.main()
