import importlib.util
import json
from pathlib import Path

spec = importlib.util.spec_from_file_location('bench', Path(__file__).parents[1] / 'scripts/rf_bench_eval.py')
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


def test_action_budget():
    assert bench.evaluation_budget({'robofollow_evaluation': {'actions_per_step': 32, 'max_steps': 16, 'action_budget': 500}}) == (32, 16)


def test_incomplete_results_are_not_success(tmp_path):
    (tmp_path / 'results.json').write_text(json.dumps([{'round': 1}]))
    run = {'status': 'failed', 'completed_trials': 1, 'expected_trials': 2}
    (tmp_path / 'run.json').write_text(json.dumps(run))
    assert not bench.complete_result(tmp_path, 2)
    run.update(status='complete', completed_trials=2)
    (tmp_path / 'run.json').write_text(json.dumps(run))
    assert not bench.complete_result(tmp_path, 2)
    (tmp_path / 'results.json').write_text(json.dumps([{'round': 1}, {'round': 1}]))
    assert not bench.complete_result(tmp_path, 2)
    (tmp_path / 'results.json').write_text(json.dumps([{'round': 1}, {'round': 2}]))
    assert bench.complete_result(tmp_path, 2)
