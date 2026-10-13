import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class A6000ExitTest(unittest.TestCase):
    def run_sim(self, fail):
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory) / 'robofollow'
            package.mkdir()
            (package / '__init__.py').touch()
            (package / 'evaluate.py').write_text(
                'import atexit, os\n'
                'def main():\n'
                '    atexit.register(lambda: print("cleanup"))\n'
                '    if os.environ["FAIL"] == "1": raise RuntimeError("evaluation failed")\n'
                '    print("evaluation")\n')
            return subprocess.run(
                [sys.executable, str(Path(__file__).resolve().parents[1] / 'scripts/rf_a6000_sim.py')],
                env=dict(os.environ, PYTHONPATH=directory, FAIL=str(int(fail))),
                capture_output=True, text=True)

    def test_success_runs_python_cleanup_and_flushes_output(self):
        result = self.run_sim(False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, 'evaluation\ncleanup\n')

    def test_evaluation_errors_remain_failures(self):
        result = self.run_sim(True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('evaluation failed', result.stderr)


if __name__ == '__main__':
    unittest.main()
