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
