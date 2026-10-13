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


def test_wait_server_checks_protocol(tmp_path):
    import socketserver
    import threading
    from types import SimpleNamespace
    import pytest
    from robofollow.serve import Server, Handler

    proc = SimpleNamespace(poll=lambda: None)
    with Server(('127.0.0.1', 0), Handler) as server:
        server.policy = SimpleNamespace(reset=lambda: None)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        log = tmp_path / 'server.log'
        port = server.server_address[1]
        log.write_text(f'Policy server: 127.0.0.1:{port}\n')
        try:
            assert bench.wait_server(proc, log, timeout=2) == port
        finally:
            server.shutdown(); thread.join()

    class HTTPHandler(socketserver.BaseRequestHandler):
        def handle(self):
            self.request.recv(4096)
            self.request.sendall(b'HTTP/1.1 400 Bad Request\r\n\r\n')

    with Server(('127.0.0.1', 0), HTTPHandler) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        log.write_text(f'Policy server: 127.0.0.1:{server.server_address[1]}\n')
        try:
            with pytest.raises(RuntimeError, match='handshake'):
                bench.wait_server(proc, log, timeout=2)
        finally:
            server.shutdown(); thread.join()
    with pytest.raises(RuntimeError, match='exited'):
        bench.wait_server(SimpleNamespace(poll=lambda: 1, returncode=1), log, timeout=2)


def test_resume_preserves_logs_and_rejects_different_checkpoint(tmp_path):
    import pytest
    meta = dict(checkpoint='final', config='cfg', stats='s', manifest='m', planner_mode='disabled',
                modes='correct,shuffle', rounds=10, base_seed=42, max_steps=16,
                actions_per_step=32, scenes='scene1', levels='L0', limit=0, tasks=1)
    (tmp_path / 'run.json').write_text(json.dumps(meta))
    (tmp_path / 'logs').mkdir()
    (tmp_path / 'logs/old.log').write_text('evidence')
    with pytest.raises(ValueError, match='checkpoint'):
        bench.prepare_run(tmp_path, dict(meta, checkpoint='other'))
    assert (tmp_path / 'logs/old.log').read_text() == 'evidence'
    bench.prepare_run(tmp_path, meta)
    assert list(tmp_path.glob('logs.previous-*/old.log'))[0].read_text() == 'evidence'
    assert list(tmp_path.glob('run.previous-*.json'))


def test_sequential_modes_finish_correct_before_shuffle(tmp_path, monkeypatch):
    import queue
    import sys
    import yaml
    from ge_act.experiments import robofollow_loading
    config = tmp_path / 'config.yaml'
    config.write_text(yaml.safe_dump(dict(
        data={'train': {'stat_file': 'stats', 'manifest_path': 'manifest'}},
        robofollow_evaluation={'actions_per_step': 32, 'max_steps': 16, 'action_budget': 500})))
    out = tmp_path / 'out'
    monkeypatch.setattr(sys, 'argv', ['bench', '--config', str(config), '--checkpoint', 'checkpoint',
                                    '--out', str(out), '--rounds', '1', '--gpus', '1',
                                    '--slots-per-gpu', '2', '--modes', 'correct,shuffle', '--sequential-modes'])
    monkeypatch.setattr(robofollow_loading, 'semantic_contract', lambda *a: {})
    monkeypatch.setattr(robofollow_loading, 'write_contract', lambda *a: None)
    monkeypatch.setattr(bench, 'list_tasks', lambda *a: [('scene1', 'L0_task')])
    sleep = bench.time.sleep
    monkeypatch.setattr(bench.time, 'sleep', lambda _: sleep(0.001))
    seen = []

    def worker(slot, gpu, mode, args, tasks, output, failures, lock):
        try:
            scene, task = tasks.get_nowait()
        except queue.Empty:
            return
        with lock:
            seen.append((mode, bench.complete_result(output / 'correct' / scene / task, 1)))
            dest = output / mode / scene / task
            dest.mkdir(parents=True)
            (dest / 'run.json').write_text(json.dumps(dict(status='complete', completed_trials=1, expected_trials=1)))
            (dest / 'results.json').write_text(json.dumps([dict(round=1, task=task,
                intent_score=1, exec_score=1, completion_ok=True)]))

    monkeypatch.setattr(bench, 'worker', worker)
    bench.main()
    assert seen == [('correct', False), ('shuffle', True)]
    assert json.loads((out / 'summary.json').read_text())['status'] == 'complete'
    assert (out / 'partial-summary.json').exists()
