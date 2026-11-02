"""Collision samples for rigid equipment mounted on named robot links.

Uses the same declared boxes as the mesh body mask, but poses them by their
own links rather than by the TCP. No grasp-target exemption applies to a
camera or its holder. Opt-in so existing rig contracts remain unchanged.
"""

from dataclasses import dataclass

import numpy as np

from .tool_volume import box_grid


@dataclass(frozen=True)
class LinkAttachments:
    parts: tuple

    def points(self, kin, q):
        frames = kin.frame_poses(q, [p[1] for p in self.parts])
        return [
            (name, local @ frame[:3, :3].T + frame[:3, 3])
            for (name, _, local), frame in zip(self.parts, frames)
        ]


def build_link_attachments(acfg, kin):
    enabled = acfg.get("check_link_attachments", False)
    if not isinstance(enabled, bool):
        raise ValueError("check_link_attachments must be true or false")
    if not enabled:
        return None
    from ..perception.robot_mask import body_mask_settings

    settings = body_mask_settings(acfg)
    if settings is None or not settings["attachments"]:
        raise ValueError("check_link_attachments requires mesh body-mask attachment boxes")
    parts = []
    for name, link, size, transform in settings["attachments"]:
        if np.max(size) > 0.5:
            raise ValueError("Link attachment box exceeds 0.5 m; check its units")
        # An extra centimetre surrounds the modeled solid. The occupancy
        # clearance margin is enforced in addition to this geometric padding.
        points = box_grid(transform[:3, 3], transform[:3, :3],
                          np.asarray(size) / 2 + 0.01, 0.02)
        parts.append((name, link, points))
    kin.frame_poses(np.zeros(kin.n), [p[1] for p in parts])  # validate links before torque
    return LinkAttachments(tuple(parts))
