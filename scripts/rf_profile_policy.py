"""GPU parity and warm-inference timing for the RoboFollow text cache."""
import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from ge_act.experiments.robofollow_policy import make_policy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    rng = np.random.default_rng(42)
    obs = {'observation': {cam: {'rgb': rng.integers(0, 256, (240, 320, 3), dtype=np.uint8)}
                          for cam in ('head_camera', 'left_camera', 'right_camera')},
           'joint_action': {'vector': np.zeros(14, dtype=np.float32)}}
    reference, timings = [], {}
    for cached in (False, True):
        policy = make_policy(args.config, args.checkpoint, config['data']['train']['stat_file'],
                             cache_text=cached)
        elapsed, index = [], 0
        for instruction in ('pick up the yellow block', 'pick up the red block'):
            policy.reset()
            policy.set_instruction(instruction)
            for repeat in range(3):
                torch.cuda.synchronize()
                started = time.perf_counter()
                action = policy.predict(obs)
                seconds = time.perf_counter() - started
                elapsed.append(dict(instruction=instruction, repeat=repeat, seconds=seconds))
                if cached:
                    np.testing.assert_array_equal(action, reference[index])
                else:
                    reference.append(action.copy())
                index += 1
                print(json.dumps(dict(cached=cached, repeat=repeat, seconds=seconds)), flush=True)
        timings[str(cached)] = elapsed
        policy.close()
        del policy
        gc.collect()
        torch.cuda.empty_cache()
    result = dict(exact_action_parity=True, timings=timings,
                  note='Synthetic fixed observations; GPU shared with baseline; not end-to-end throughput.')
    Path(args.out).write_text(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
