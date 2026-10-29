"""Plate scene and independent placement audit for the B601-RS demo."""
from pathlib import Path
import math
import xml.etree.ElementTree as ET

import numpy as np

PLATE_XY = (0.23, -0.14)
PLATE_RADIUS = 0.085
PLATE_TOP = 0.008


def prepare(cfg, run_dir):
    from cascade.sim.demo_scene import write_demo_scene
    # Include the robot alone: the generated wrapper supplies the single floor.
    source = Path(cfg.arm.mjcf).with_name('seeed_rebot_devarm.xml')
    generated = write_demo_scene(source, cfg.camera, cameras=cfg.cameras)
    tree = ET.parse(generated)
    body = ET.SubElement(tree.getroot().find('worldbody'), 'body',
                         name='blue_plate', pos=f'{PLATE_XY[0]} {PLATE_XY[1]} 0')
    ET.SubElement(body, 'geom', name='plate_base', type='cylinder',
                  size=f'{PLATE_RADIUS} {PLATE_TOP / 2}', pos=f'0 0 {PLATE_TOP / 2}',
                  rgba='0.10 0.20 0.80 1', friction='1 .03 .003')
    for i in range(32):
        a, b = i * 2 * math.pi / 32, (i + 1) * 2 * math.pi / 32
        points = [PLATE_RADIUS*math.cos(a), PLATE_RADIUS*math.sin(a), PLATE_TOP,
                  PLATE_RADIUS*math.cos(b), PLATE_RADIUS*math.sin(b), PLATE_TOP]
        ET.SubElement(body, 'geom', name=f'plate_rim_{i}', type='capsule', size='.002',
                      fromto=' '.join(map(str, points)), rgba='0.10 0.20 0.80 1')
    scene = source.with_name(f'{run_dir.name}-plate.xml')
    tree.write(scene, encoding='unicode')
    for arm_cfg in [cfg.arm, *cfg.arms]:
        data = arm_cfg._data if hasattr(arm_cfg, '_data') else arm_cfg
        data['mjcf'] = str(scene)
        data['mj_prop_from_camera'] = False
    for camera in [cfg.camera, *cfg.cameras]:
        data = camera._data if hasattr(camera, '_data') else camera
        data['mj_scene'] = str(scene)
        data.pop('mj_scene_source', None)
        detector = data['detector']
        detector = detector._data if hasattr(detector, '_data') else detector
        detector['extra_labels'] = ['blue plate']
    return scene


def audit(world, arm, gripper):
    """Read native collision contacts and cube corners; no LLM/belief claims."""
    import mujoco
    m, d = world.model, world.data
    with world.lock:
        mujoco.mj_forward(m, d)
        gid = m.geom('red_cube').id
        plate = m.body('blue_plate').id
        R = d.geom_xmat[gid].reshape(3, 3)
        center = d.geom_xpos[gid].copy()
        corners = np.array([[x,y,z] for x in [-1,1] for y in [-1,1] for z in [-1,1]])
        corners = (corners * m.geom_size[gid]) @ R.T + center
        max_radius = float(np.linalg.norm(corners[:, :2] - PLATE_XY, axis=1).max())
        support_force = 0.
        robot_contact = False
        for contact_index in range(d.ncon):
            contact = d.contact[contact_index]
            if gid not in (contact.geom1, contact.geom2):
                continue
            other = contact.geom2 if contact.geom1 == gid else contact.geom1
            other_body = int(m.geom_bodyid[other])
            force = np.zeros(6)
            mujoco.mj_contactForce(m, d, contact_index, force)
            if other_body == plate:
                support_force += max(0., float(force[0]))
            elif other_body != 0 and force[0] > 1e-4:
                robot_contact = True
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY,
                                m.body('red_cube').id, velocity, 0)
        state = arm.get_state()
        opened = (state.gripper_pos - gripper.closed_pos) / (gripper.open_pos - gripper.closed_pos)
        checks = {
            'footprint_inside_plate': max_radius < PLATE_RADIUS - .002,
            'bottom_at_plate_surface': abs(float(corners[:, 2].min()) - PLATE_TOP) < .003,
            'plate_support_contact': support_force > .01,
            'no_robot_contact': not robot_contact,
            'gripper_open': float(opened) > .9,
            'cube_at_rest': np.linalg.norm(velocity[3:]) < .01 and np.linalg.norm(velocity[:3]) < .1,
        }
        return {'passed': all(bool(v) for v in checks.values()),
                'checks': {k: bool(v) for k,v in checks.items()},
                'cube_center': center.tolist(), 'max_corner_radius_m': max_radius,
                'cube_bottom_m': float(corners[:, 2].min()),
                'plate_support_force_n': support_force, 'gripper_open_fraction': float(opened),
                'cube_velocity_angular_linear': velocity.tolist()}
