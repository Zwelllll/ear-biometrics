"""Synthetic smoke-test scores must never enter experiment results."""
from contextlib import redirect_stdout
from dataclasses import replace
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import smoke_test
from src import results


class SmokeResultsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.paths = replace(results.P, results=self.root / 'production')
        self.patcher = patch.object(results, 'P', self.paths)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)
        device = patch.object(results, '_device_name', return_value='test-cpu')
        device.start()
        self.addCleanup(device.stop)

    def smoke(self):
        with redirect_stdout(io.StringIO()):
            smoke_test.check_results_logging()

    def test_repeated_smoke_preserves_real_results_byte_for_byte(self):
        results.log_run(train_dataset='ami', model='resnet50', arm='zoom+aug',
                        seed=0, test_top1=0.75)
        before = self.paths.results_csv.read_bytes()
        self.smoke()
        self.smoke()
        self.assertEqual(self.paths.results_csv.read_bytes(), before)
        self.assertEqual(len(results.load_results()), 1)
        self.assertEqual(results.summarise().iloc[0]['mean'], 0.75)

    def test_smoke_does_not_create_production_results(self):
        self.smoke()
        self.assertFalse(self.paths.results_csv.exists())

    def test_temporary_results_removed_on_success(self):
        with patch.object(results, 'log_run', wraps=results.log_run) as log:
            self.smoke()
        paths = {call.kwargs['path'] for call in log.call_args_list}
        self.assertEqual(len(paths), 1)
        for path in paths:
            self.assertNotEqual(path, self.paths.results_csv)
            self.assertFalse(path.parent.exists())

    def test_failure_cleans_temporary_storage_and_preserves_results(self):
        results.log_run(train_dataset='ami', test_top1=0.8)
        before = self.paths.results_csv.read_bytes()
        with patch.object(results, 'log_run', wraps=results.log_run) as log:
            with patch.object(results, 'summarise', side_effect=RuntimeError('summary failed')):
                with self.assertRaisesRegex(RuntimeError, 'summary failed'):
                    self.smoke()
        self.assertEqual(self.paths.results_csv.read_bytes(), before)
        for call in log.call_args_list:
            self.assertFalse(call.kwargs['path'].parent.exists())

    def test_explicit_path_round_trip_and_default_path_still_work(self):
        isolated = self.root / 'isolated/results.csv'
        self.assertTrue(results.load_results(path=isolated).empty)
        results.log_run(path=isolated, train_dataset='ami', model='resnet50',
                        arm='raw', seed=0, test_top1=0.5)
        self.assertEqual(results.load_results(path=isolated).iloc[0]['test_top1'], 0.5)
        self.assertEqual(results.summarise(path=isolated).iloc[0]['mean'], 0.5)
        self.assertFalse(self.paths.results_csv.exists())
        results.log_run(train_dataset='ami', test_top1=0.9)
        self.assertEqual(results.load_results().iloc[0]['test_top1'], 0.9)
        with self.assertRaises(KeyError):
            results.log_run(path=isolated, test_topl=1.0)


if __name__ == '__main__':
    unittest.main()
