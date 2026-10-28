"""Print saved segmentation validation metrics; never run evaluation or training."""
import argparse
import json
import math
from pathlib import Path

METRICS = [('BABEL120-mAP (%)', 'babel-120', 'frame_map'),
           ('BABEL60-mAP (%)', 'babel-60', 'frame_map'),
           ('Micro-F1 (%)', 'babel-120', 'micro_f1'),
           ('Macro-F1 (%)', 'babel-120', 'macro_f1'),
           ('F1@10 (%)', 'babel-120', 'segment_f1_10'),
           ('F1@25 (%)', 'babel-120', 'segment_f1_25'),
           ('F1@50 (%)', 'babel-120', 'segment_f1_50')]


def read_json(path):
    return json.loads(path.read_text())


def load_report(path, selection=None):
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise ValueError(f'Input does not exist: {path}')
    if path.is_dir():
        candidates = [path] if (path / 'metrics.json').is_file() else sorted(
            {p.parent for p in path.rglob('metrics.json')})
        if len(candidates) != 1:
            raise ValueError('Expected one experiment; specify a run directory. Candidates:\n' +
                             '\n'.join(map(str, candidates)))
        root = candidates[0]
        history = read_json(root / 'metrics.json')
        summary = read_json(root / 'summary.json') if (root / 'summary.json').exists() else {}
        config = read_json(root / 'config.json') if (root / 'config.json').exists() else {}
        if not history:
            raise ValueError('No completed validation evaluations are saved yet')
        if selection == 'latest':
            step = history[-1]['iteration']
        elif selection and selection != 'best':
            step = int(selection)
        else:
            step = summary.get('best_step')
            if step is None:
                scored = [r for r in history if r.get('val', {}).get('babel-120', {}).get('frame_map') is not None]
                if not scored:
                    raise ValueError('No validation mAP available for best selection')
                step = max(scored, key=lambda r: r['val']['babel-120']['frame_map'])['iteration']
        params = summary.get('parameters', {}).get('total')
        complete = bool(summary.get('iterations_completed'))
        source = str(root / 'metrics.json')
    else:
        if selection is not None:
            raise ValueError('--iteration applies to experiment directories only; checkpoint input uses its own step')
        from experiment.utils import _torch_load_checkpoint
        checkpoint = _torch_load_checkpoint(path)
        history = checkpoint.get('history', [])
        step = checkpoint.get('global_step')
        config = checkpoint.get('signature', {})
        # A state_dict includes buffers and may include frozen JEPA; do not count it as trainable.
        params = None
        complete = None
        source = str(path)
    matches = [r for r in history if r.get('iteration') == step]
    if len(matches) != 1 or 'val' not in matches[0]:
        raise ValueError(f'No unique saved validation evaluation for iteration {step}; cannot substitute another checkpoint')
    val = matches[0]['val']
    values = {}
    for label, group, key in METRICS:
        value = val.get(group, {}).get(key)
        if value is not None and (not isinstance(value, (int, float)) or not math.isfinite(value)):
            raise ValueError(f'Invalid metric: {label}')
        values[label] = None if value is None else value * 100
    return dict(source=source, iteration=step, split='validation', completed=complete,
                precision='BF16 mixed' if config.get('optimization', {}).get('use_bfloat16', False) else 'FP32',
                trainable_parameters=params, metrics=values,
                note='Saved validation metrics; selection and reporting share validation. Missing metrics are N/A, not zero.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiment_or_checkpoint', type=Path)
    parser.add_argument('--iteration', help='Directory only: best (default), latest, or an evaluated iteration')
    parser.add_argument('--json', action='store_true', help='Print machine-readable JSON; metric values are percentage points')
    args = parser.parse_args()
    try:
        result = load_report(args.experiment_or_checkpoint, args.iteration)
    except (ValueError, KeyError, OSError) as error:
        parser.error(str(error))
    if args.json:
        print(json.dumps(result, indent=2, allow_nan=False))
        return
    print(f"Source: {result['source']}\nSplit: validation | Iteration: {result['iteration']} | Precision: {result['precision']}")
    if result['completed'] is False:
        print('Status: in progress (best saved validation so far)')
    for label, value in result['metrics'].items():
        print(f'{label:20s} {value:.2f}' if value is not None else f'{label:20s} N/A')
    print('Trainable parameters:', result['trainable_parameters'] if result['trainable_parameters'] is not None else 'N/A')
    print(result['note'])


if __name__ == '__main__':
    main()
