"""Configuration, task contract and cleanup for this single-scene experiment."""
import json

from cascade.lifecycle import teardown_receipt, teardown_step

LIFT_TASK = 'Pick up the red cube and hold it above the table.'
PLATE_TASK = ('Pick up the red cube and place it on the blue plate. '
              'Leave the cube resting on the plate with the gripper clear of it.')


def configure():
    from cascade.config import load_demo_config

    cfg = load_demo_config(camera='rebot_mujoco_scene', arm='rebot_rs_mujoco',
                           llm='inferencehub_astra', _ignore_robot_environment=True)
    # Explicit demo choices, not defaults for the robot or Astra profiles.
    cfg.grasp._data.update(backend='obb', max_pick_attempts=2, persist_seconds=60)
    cfg.occupancy._data['enabled'] = False
    return cfg


def task_contract(args, parser):
    if args.task is not None and args.verify is None and not args.inspect_only:
        parser.error('--task requires --verify lift, plate, or none')
    mode = args.verify or ('plate' if args.plate else 'lift')
    if mode == 'plate' and not args.plate:
        parser.error('--verify plate requires --plate')
    return args.task or (PLATE_TASK if args.plate else LIFT_TASK), mode


def verdict(mode, model_success, lift, placement):
    independent = (lift > 0.03 if mode == 'lift' else
                   placement['passed'] if mode == 'plate' else None)
    return {'mode': mode, 'independent_success': independent,
            'success': bool(model_success and independent is not False)}


def cleanup(output, *, recorder=None, simulation=None, runtime=None, arm=None, llm=None):
    from cascade.apps.demo import shutdown_runtime

    stages = []
    for name, close in (
        ('recorder', recorder.stop if recorder is not None else None),
        ('desktop', simulation.close if simulation is not None else None),
        ('runtime', (lambda: shutdown_runtime(runtime, arm)) if runtime is not None else None),
        ('llm', llm.close if llm is not None else None),
    ):
        if close is not None:
            stages.append(teardown_step(name, close))
    receipt = teardown_receipt(stages)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'teardown.json').write_text(json.dumps(receipt, indent=2))
    return receipt
