"""Notebook checks that require neither datasets nor a GPU.

Run with: python -m unittest discover -s tests -v
"""
import ast
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


NOTEBOOK = Path(__file__).resolve().parents[1] / 'notebooks/kaggle_phase4_ami.ipynb'


class KaggleNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = json.loads(NOTEBOOK.read_text(encoding='utf-8'))
        cls.code = [''.join(c['source']) for c in cls.notebook['cells']
                    if c['cell_type'] == 'code']
        cls.setup = next(s for s in cls.code if 'repo_url =' in s)

    def test_all_code_cells_compile(self):
        for i, source in enumerate(self.code):
            with self.subTest(cell=i):
                compile(source, f'notebook-cell-{i}', 'exec')

    def test_external_commands_fail_loudly(self):
        for source in self.code:
            for node in ast.walk(ast.parse(source)):
                if (isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Attribute)
                        and isinstance(node.func.value, ast.Name)
                        and node.func.value.id == 'subprocess'
                        and node.func.attr == 'run'):
                    self.assertTrue(any(k.arg == 'check' and
                                        isinstance(k.value, ast.Constant) and
                                        k.value.value is True for k in node.keywords))

    def run_setup(self, directory):
        source = self.setup.replace('"/kaggle/working/ear-biometrics"', repr(str(directory)))
        exec(compile(source, 'setup-cell', 'exec'), {})

    def test_rerun_preserves_existing_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / 'ear-biometrics'
            (repo / '.git').mkdir(parents=True)
            artifacts = [repo / 'results/results.csv', repo / 'checkpoints/latest.pth']
            for artifact in artifacts:
                artifact.parent.mkdir(exist_ok=True)
                artifact.write_bytes(b'previous experiment')
            with patch('subprocess.run') as run, patch('os.chdir'):
                self.run_setup(repo)
                self.run_setup(repo)
            self.assertEqual(run.call_count, 2)
            for call in run.call_args_list:
                self.assertEqual(call.args[0], ['git', 'log', '--oneline', '-1'])
            for artifact in artifacts:
                self.assertEqual(artifact.read_bytes(), b'previous experiment')

    def test_new_directory_is_cloned(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / 'ear-biometrics'
            with patch('subprocess.run') as run, patch('os.chdir'):
                self.run_setup(repo)
            self.assertEqual(run.call_args_list[0].args[0], [
                'git', 'clone', '--quiet',
                'https://github.com/Zwelllll/ear-biometrics.git', str(repo)])

    def test_existing_non_checkout_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            artifact = repo / 'results.csv'
            artifact.write_bytes(b'saved results')
            with patch('subprocess.run') as run, patch('os.chdir'), self.assertRaises(RuntimeError):
                self.run_setup(repo)
            run.assert_not_called()
            self.assertEqual(artifact.read_bytes(), b'saved results')

    def test_clone_failure_stops_setup(self):
        import subprocess
        with tempfile.TemporaryDirectory() as tmp:
            with patch('subprocess.run', side_effect=subprocess.CalledProcessError(1, 'git')), patch('os.chdir') as chdir:
                with self.assertRaises(subprocess.CalledProcessError):
                    self.run_setup(Path(tmp) / 'missing')
                chdir.assert_not_called()


if __name__ == '__main__':
    unittest.main()
