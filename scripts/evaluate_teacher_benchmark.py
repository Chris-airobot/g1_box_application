from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import tyro

from holosoma.config_types.command import MotionConfig
from holosoma.config_types.experiment import ExperimentConfig
from holosoma.managers.command.terms.wbt import _resolve_motion_files
from holosoma.utils.eval_utils import (
    CheckpointConfig,
    init_eval_logging,
    load_checkpoint,
    load_saved_experiment_config,
)
from holosoma.utils.helpers import get_class
from holosoma.utils.safe_torch_import import torch
from holosoma.utils.sim_utils import close_simulation_app, setup_simulation_environment
from holosoma.utils.tyro_utils import TYRO_CONIFG


def wilson_interval(k: int, n: int, z: float = 1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    den = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / den
    half = z * math.sqrt((p * (1.0 - p) / n) + z * z / (4.0 * n * n)) / den
    return center - half, center + half


def main():
    init_eval_logging()

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--trials-per-motion', type=int, default=20)
    parser.add_argument('--max-steps', type=int, default=100000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', default=None)
    bench, remaining = parser.parse_known_args()

    random.seed(bench.seed)
    np.random.seed(bench.seed)
    torch.manual_seed(bench.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(bench.seed)

    checkpoint_cfg = CheckpointConfig(checkpoint=bench.checkpoint)
    saved_cfg, _ = load_saved_experiment_config(checkpoint_cfg)
    eval_cfg = saved_cfg.get_eval_config()

    # Same compatibility conversion used by eval_agent.py.
    if eval_cfg.command is not None:
        motion_term = eval_cfg.command.setup_terms.get('motion_command')
        if motion_term is not None:
            motion_cfg = motion_term.params.get('motion_config')
            if isinstance(motion_cfg, dict):
                motion_term.params['motion_config'] = MotionConfig(**motion_cfg)

    cfg = tyro.cli(
        ExperimentConfig,
        default=eval_cfg,
        args=remaining,
        description='Teacher benchmark config overrides',
        config=TYRO_CONIFG,
    )

    ckpt_path = Path(bench.checkpoint).expanduser()
    if bench.output_dir is None:
        out = ckpt_path.parent / 'benchmark_all_motions'
    else:
        out = Path(bench.output_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    print('=== BENCHMARK CONFIG ===')
    print('checkpoint:', ckpt_path)
    print('trials per motion:', bench.trials_per_motion)
    print('max sim steps:', bench.max_steps)
    print('seed:', bench.seed)
    print('output:', out.resolve())

    env = None
    simulation_app = None

    try:
        env, device, simulation_app = setup_simulation_environment(cfg)

        runtime_dir = out / 'runtime'
        runtime_dir.mkdir(parents=True, exist_ok=True)
        loaded_ckpt = load_checkpoint(bench.checkpoint, str(runtime_dir))

        algo_class = get_class(cfg.algo._target_)
        algo = algo_class(
            device=device,
            env=env,
            config=cfg.algo.config,
            log_dir=str(runtime_dir),
            multi_gpu_cfg=None,
        )
        algo.setup()
        algo.load(str(loaded_ckpt))
        algo._eval_mode()

        env.set_is_evaluating()
        obs = env.reset_all()
        policy = algo.get_inference_policy()

        motion_cmd = env.command_manager.get_state('motion_command')
        if motion_cmd is None:
            raise RuntimeError('motion_command state not found')

        motion_files = _resolve_motion_files(motion_cmd.motion_cfg)
        num_motions = len(motion_cmd.motions)
        if len(motion_files) != num_motions:
            raise RuntimeError(
                f'motion file count mismatch: files={len(motion_files)} loaded={num_motions}'
            )

        names = [Path(p).name for p in motion_files]
        quota = int(bench.trials_per_motion)
        target_total = num_motions * quota

        print('\n=== LOADED MOTIONS ===')
        print('num motions:', num_motions)
        print('target accepted trials:', target_total)

        counts = [0 for _ in range(num_motions)]
        rows = []

        ep_return = torch.zeros(env.num_envs, device=device, dtype=torch.float32)
        ep_len = torch.zeros(env.num_envs, device=device, dtype=torch.long)

        tracking_keys = [
            'motion/error_ref_pos',
            'motion/error_ref_rot',
            'motion/error_body_pos',
            'motion/error_body_rot',
            'motion/error_joint_pos',
            'motion/error_joint_vel',
        ]
        metric_sum = defaultdict(float)
        metric_count = defaultdict(int)

        accepted = 0
        last_report = -1

        for sim_step in range(int(bench.max_steps)):
            # Save episode identity BEFORE env.step(), because done envs are reset
            # and assigned a new random motion inside env.step().
            prev_motion_ids = motion_cmd.motion_ids.clone()
            prev_time_steps = motion_cmd.time_steps.clone()

            actor_obs = torch.cat([obs[k] for k in algo.actor_obs_keys], dim=1)
            actions = policy({'actor_obs': actor_obs})

            obs, rewards, dones, extras = env.step({'actions': actions})

            ep_return += rewards
            ep_len += 1

            # Aggregate tracking diagnostics over evaluation steps.
            to_log = extras.get('to_log', {}) if isinstance(extras, dict) else {}
            for key in tracking_keys:
                if key not in to_log:
                    continue
                value = to_log[key]
                if isinstance(value, torch.Tensor):
                    value = float(value.float().mean().item())
                else:
                    try:
                        value = float(value)
                    except Exception:
                        continue
                if math.isfinite(value):
                    metric_sum[key] += value
                    metric_count[key] += 1

            done_ids = dones.nonzero(as_tuple=False).flatten()
            if done_ids.numel() == 0:
                continue

            term_results = env.termination_manager.last_term_results

            for env_id_t in done_ids:
                env_id = int(env_id_t.item())
                motion_id = int(prev_motion_ids[env_id].item())

                # We only keep the first N trials for each motion so every motion
                # contributes exactly the same amount to the final statistics.
                if counts[motion_id] < quota:
                    reasons = []
                    for name, result in term_results.items():
                        if bool(result[env_id].item()):
                            reasons.append(name)

                    # If bad_tracking + bad_tracking/body_pos both fire, keep the
                    # specific child reason, matching BaseTask's log behavior.
                    parent_terms = {
                        name.split('/', 1)[0]
                        for name in reasons
                        if '/' in name
                    }
                    if parent_terms:
                        reasons = [r for r in reasons if r not in parent_terms]

                    if not reasons:
                        reasons = ['unknown']

                    non_success = [r for r in reasons if r != 'motion_ends']
                    success = ('motion_ends' in reasons) and (len(non_success) == 0)
                    primary_reason = 'motion_ends' if success else '+'.join(sorted(non_success or reasons))

                    length_ref = max(int(motion_cmd.motion_lengths[motion_id].item()) - 1, 1)
                    progress = min(
                        max((int(prev_time_steps[env_id].item()) + 1) / length_ref, 0.0),
                        1.0,
                    )

                    trial_index = counts[motion_id] + 1
                    row = {
                        'motion_id': motion_id,
                        'motion_file': names[motion_id],
                        'trial': trial_index,
                        'success': int(success),
                        'reason': primary_reason,
                        'episode_return': float(ep_return[env_id].item()),
                        'episode_steps': int(ep_len[env_id].item()),
                        'reference_progress': float(progress),
                        'terminal_reference_step': int(prev_time_steps[env_id].item()) + 1,
                        'reference_length': int(motion_cmd.motion_lengths[motion_id].item()),
                    }
                    rows.append(row)
                    counts[motion_id] += 1
                    accepted += 1

                # New episode has already been created by env.step().
                ep_return[env_id] = 0.0
                ep_len[env_id] = 0

            if hasattr(algo.actor, 'reset'):
                algo.actor.reset(dones)

            pct = int(100 * accepted / target_total)
            if pct >= last_report + 5:
                last_report = pct
                print(
                    f'progress: {accepted}/{target_total} ({pct}%) | '
                    f'min trials/motion={min(counts)} max={max(counts)}'
                )

            if min(counts) >= quota:
                print(f'Coverage complete at sim step {sim_step}.')
                break
        else:
            print('WARNING: max_steps reached before equal per-motion coverage completed')

        # ------------------------------------------------------------------
        # Save trial-level CSV
        # ------------------------------------------------------------------
        trial_csv = out / 'trials.csv'
        fieldnames = [
            'motion_id',
            'motion_file',
            'trial',
            'success',
            'reason',
            'episode_return',
            'episode_steps',
            'reference_progress',
            'terminal_reference_step',
            'reference_length',
        ]
        with trial_csv.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        # ------------------------------------------------------------------
        # Per-motion summary
        # ------------------------------------------------------------------
        per_motion_rows = []
        for motion_id in range(num_motions):
            rr = [r for r in rows if r['motion_id'] == motion_id]
            n = len(rr)
            k = sum(int(r['success']) for r in rr)
            lo, hi = wilson_interval(k, n)
            reasons = Counter(r['reason'] for r in rr if not int(r['success']))
            failed_progress = [float(r['reference_progress']) for r in rr if not int(r['success'])]

            per_motion_rows.append(
                {
                    'motion_id': motion_id,
                    'motion_file': names[motion_id],
                    'trials': n,
                    'successes': k,
                    'completion_rate': (k / n) if n else float('nan'),
                    'wilson95_low': lo,
                    'wilson95_high': hi,
                    'mean_episode_return': float(np.mean([r['episode_return'] for r in rr])) if rr else float('nan'),
                    'mean_episode_steps': float(np.mean([r['episode_steps'] for r in rr])) if rr else float('nan'),
                    'mean_failure_progress': float(np.mean(failed_progress)) if failed_progress else float('nan'),
                    'failure_reasons': json.dumps(dict(reasons), sort_keys=True),
                }
            )

        per_motion_csv = out / 'per_motion.csv'
        with per_motion_csv.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(per_motion_rows[0].keys()))
            writer.writeheader()
            writer.writerows(per_motion_rows)

        # ------------------------------------------------------------------
        # Overall summary
        # ------------------------------------------------------------------
        n = len(rows)
        k = sum(int(r['success']) for r in rows)
        lo, hi = wilson_interval(k, n)
        failure_counts = Counter(r['reason'] for r in rows if not int(r['success']))
        failed_progress = [float(r['reference_progress']) for r in rows if not int(r['success'])]

        aggregate_tracking = {
            key: metric_sum[key] / metric_count[key]
            for key in tracking_keys
            if metric_count[key] > 0
        }

        summary = {
            'checkpoint': str(ckpt_path),
            'seed': bench.seed,
            'num_motions': num_motions,
            'trials_per_motion_target': quota,
            'accepted_trials': n,
            'successes': k,
            'motion_completion_rate': (k / n) if n else 0.0,
            'wilson95_low': lo,
            'wilson95_high': hi,
            'failure_counts': dict(failure_counts),
            'mean_episode_return': float(np.mean([r['episode_return'] for r in rows])) if rows else float('nan'),
            'mean_episode_steps': float(np.mean([r['episode_steps'] for r in rows])) if rows else float('nan'),
            'mean_failure_progress': float(np.mean(failed_progress)) if failed_progress else None,
            'aggregate_tracking_metrics': aggregate_tracking,
            'coverage_per_motion': counts,
        }

        summary_json = out / 'summary.json'
        summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')

        tracking_csv = out / 'aggregate_tracking_metrics.csv'
        with tracking_csv.open('w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['metric', 'mean'])
            for key, value in aggregate_tracking.items():
                writer.writerow([key, value])

        print('\n=== BENCHMARK RESULT ===')
        print(f'motions:                  {num_motions}')
        print(f'accepted trials:          {n}')
        print(f'successes / motion_ends:  {k}')
        print(f'completion rate:          {100.0 * summary["motion_completion_rate"]:.2f}%')
        print(f'95% Wilson CI:            [{100.0 * lo:.2f}%, {100.0 * hi:.2f}%]')
        print(f'mean episode return:      {summary["mean_episode_return"]:.4f}')
        print(f'mean episode steps:       {summary["mean_episode_steps"]:.2f}')
        print('failure counts:           ', dict(failure_counts))
        print('aggregate tracking:       ', aggregate_tracking)
        print('coverage min/max:         ', min(counts), max(counts))

        print('\n=== WORST 10 MOTIONS BY COMPLETION RATE ===')
        for r in sorted(per_motion_rows, key=lambda x: (x['completion_rate'], x['motion_id']))[:10]:
            print(
                f"{r['motion_id']:02d}  {100*r['completion_rate']:6.2f}%  "
                f"{r['motion_file']}  failures={r['failure_reasons']}"
            )

        print('\n=== OUTPUT FILES ===')
        print(trial_csv)
        print(per_motion_csv)
        print(summary_json)
        print(tracking_csv)

        print('\nBENCHMARK_COMPLETE')

    finally:
        # All CSV/JSON files are written before shutdown. If IsaacSim teardown
        # hangs as seen previously, Ctrl+C after BENCHMARK_COMPLETE is safe.
        if simulation_app is not None:
            close_simulation_app(simulation_app)


if __name__ == '__main__':
    main()
