"""Parallel RoboFollow benchmark: one policy server per simulator slot, tasks pulled from a shared queue.

Usage: rf_bench_eval.py --config RUN.yaml --checkpoint CK.safetensors --out DIR
       [--planner-mode disabled|predicted] [--modes correct,shuffle] [--levels L0,L1,L2,L3]
       [--rounds 10] [--max-steps 10] [--slots-per-gpu 3] [--scenes scene1,...] [--limit N]
"""
import argparse, json, os, queue, socket, subprocess, sys, threading, time
from pathlib import Path

W = Path('/data/users/junjie/workspace/VLM4WAM_robofollow_q35')
RT = Path('/data/users/junjie/workspace/robofollow/RoboTwin')
D = '/data/users/junjie/workspace/hpc3_jhe724'
POLICY_PY = f'{D}/.conda/envs/qwen35/bin/python'
SIM_PY = '/data/users/junjie/workspace/robofollow/env/bin/python'
OVERLAY = f'{D}/envs/overlay_peft_tf5'
SCENES = ('scene1', 'scene2', 'scene3', 'scene4')


def list_tasks(scenes, levels):
    tasks = []
    for scene in scenes:
        out = subprocess.run([SIM_PY, '-m', 'robofollow.evaluate', '--scene', scene, '--list'], cwd=RT,
                             capture_output=True, text=True, check=True).stdout
        for line in out.splitlines():
            parts = line.split(' ', 2)
            if len(parts) >= 2 and parts[0] == scene and parts[1].rstrip(':').split('_')[0] in levels:
                tasks.append((scene, parts[1].rstrip(':')))
    return tasks


def wait_port(proc, port, timeout=1200):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if proc.poll() is not None:
            raise RuntimeError(f'policy server on {port} exited with {proc.returncode}')
        try:
            with socket.create_connection(('127.0.0.1', port), timeout=0.5):
                return
        except OSError:
            time.sleep(2)
    raise TimeoutError(f'server {port} not ready')


def worker(slot, gpu, port, mode, args, tasks, out, failures, lock):
    kwargs = dict(config_path=args.config, checkpoint_path=args.checkpoint, stats_path=args.stats,
                  manifest_path=args.manifest, planner_mode=args.planner_mode, instruction_mode=mode)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), HDF5_USE_FILE_LOCKING='FALSE', TOKENIZERS_PARALLELISM='false',
               PYTHONPATH=f'{W}:{W}/ge_act:{RT}:{OVERLAY}')
    logdir = out / 'logs'; logdir.mkdir(parents=True, exist_ok=True)
    with open(logdir / f'serve_{mode}_{slot}.log', 'w') as slog:
        server = subprocess.Popen([POLICY_PY, '-m', 'robofollow.serve', '--factory', 'ge_act.experiments.robofollow_policy:make_policy',
                                   '--kwargs', json.dumps(kwargs), '--host', '127.0.0.1', '--port', str(port)],
                                  cwd=W, env=env, stdout=slog, stderr=subprocess.STDOUT)
        try:
            wait_port(server, port)
            sim_env = dict(os.environ, CUDA_HOME='/usr/local/cuda-12.8', VK_ICD_FILENAMES='/etc/vulkan/icd.d/nvidia_icd.json')
            while True:
                try:
                    scene, task = tasks.get_nowait()
                except queue.Empty:
                    return
                dest = out / mode / scene / task
                if (dest / 'results.json').exists():
                    continue
                for attempt in range(2):
                    if dest.exists():
                        subprocess.run(['rm', '-rf', str(dest)])
                    cmd = [SIM_PY, '-m', 'robofollow.evaluate', '--scene', scene, '--tasks', task, '--rounds', str(args.rounds),
                           '--base-seed', str(args.base_seed), '--max-steps', str(args.max_steps), '--actions-per-step', '50',
                           '--runtime', 'fixed', '--sim-steps', '15', '--gpu', str(gpu), '--remote', '--host', '127.0.0.1',
                           '--port', str(port), '--output', str(dest)]
                    with open(logdir / f'sim_{mode}_{scene}_{task}.log', 'w') as elog:
                        rc = subprocess.run(cmd, cwd=RT, env=sim_env, stdout=elog, stderr=subprocess.STDOUT).returncode
                    if rc == 0 and (dest / 'results.json').exists():
                        break
                else:
                    with lock:
                        failures.append((mode, scene, task))
        finally:
            server.terminate()
            try:
                server.wait(30)
            except subprocess.TimeoutExpired:
                server.kill()


def summarize(out, modes):
    from statistics import mean
    summary = {}
    for mode in modes:
        rows = []
        for f in (out / mode).glob('*/*/results.json'):
            rows += json.loads(f.read_text())
        by = {}
        for r in rows:
            by.setdefault(r['task'].split('_')[0], []).append(r)
        by['all'] = rows
        summary[mode] = {lvl: dict(trials=len(rs), intent=mean(r['intent_score'] for r in rs), exec=mean(r['exec_score'] for r in rs),
                                   completion=mean(float(bool(r['completion_ok'])) for r in rs)) for lvl, rs in sorted(by.items()) if rs}
    if len(modes) == 2 and all(summary.get(m) for m in modes):
        summary['delta_correct_minus_shuffle'] = {lvl: {k: summary['correct'][lvl][k] - summary['shuffle'][lvl][k] for k in ('intent', 'exec', 'completion')}
                                                  for lvl in summary['correct'] if lvl in summary['shuffle']}
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True); p.add_argument('--checkpoint', required=True); p.add_argument('--out', required=True)
    p.add_argument('--planner-mode', default='disabled'); p.add_argument('--modes', default='correct')
    p.add_argument('--levels', default='L0,L1,L2,L3'); p.add_argument('--scenes', default=','.join(SCENES))
    p.add_argument('--rounds', type=int, default=10); p.add_argument('--max-steps', type=int, default=10)
    p.add_argument('--slots-per-gpu', type=int, default=3); p.add_argument('--gpus', type=int, default=0)
    p.add_argument('--limit', type=int, default=0); p.add_argument('--base-seed', type=int, default=42)
    args = p.parse_args()
    import yaml
    c = yaml.safe_load(open(args.config))
    args.stats, args.manifest = c['data']['train']['stat_file'], c['data']['train']['manifest_path']
    sys.path[:0] = [str(W), str(W / 'ge_act')]
    from ge_act.experiments.robofollow_loading import semantic_contract, write_contract
    write_contract(args.checkpoint, semantic_contract(c, args.stats), dict(capture_phase='rf_bench_eval binding', config=args.config))
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    modes = args.modes.split(',')
    tasks = list_tasks(args.scenes.split(','), set(args.levels.split(',')))
    if args.limit:
        tasks = tasks[:args.limit]
    gpus = args.gpus or int(subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True).stdout.count('GPU '))
    slots = gpus * args.slots_per_gpu
    json.dump(dict(vars(args), tasks=len(tasks), slots=slots, started=time.time()), open(out / 'run.json', 'w'), indent=1)
    print(f'[bench] {len(tasks)} tasks x {args.rounds} rounds x modes {modes}; {slots} slots on {gpus} GPUs', flush=True)
    queues = {m: queue.Queue() for m in modes}
    for m in modes:
        for t in tasks:
            queues[m].put(t)
    failures, lock, threads = [], threading.Lock(), []
    for i in range(slots):
        mode = modes[i % len(modes)]
        th = threading.Thread(target=worker, args=(i, i % gpus, 9100 + i, mode, args, queues[mode], out, failures, lock), daemon=True)
        th.start(); threads.append(th)
        time.sleep(5)
    while any(t.is_alive() for t in threads):
        time.sleep(60)
        done = {m: len(list((out / m).glob('*/*/results.json'))) for m in modes}
        print(f'[bench] {time.strftime("%H:%M:%S")} done {done} / {len(tasks)} failures {len(failures)}', flush=True)
    summary = summarize(out, modes)
    summary['failures'] = failures
    json.dump(summary, open(out / 'summary.json', 'w'), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
