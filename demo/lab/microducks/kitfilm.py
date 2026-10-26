"""[Kit] Shared pieces for re-rendering recorded Newton traces in RTX: quaternion helpers, the canonical MicroDuck posed
body by body (each rigid body relative to its parent, physics disabled) and a Replicator RGB grabber."""
from __future__ import annotations

import numpy as np


def quat_mul(a, b):
    """Hamilton product of xyzw quaternions (broadcasting)."""
    ax, ay, az, aw = np.moveaxis(np.asarray(a, float), -1, 0)
    bx, by, bz, bw = np.moveaxis(np.asarray(b, float), -1, 0)
    return np.stack((aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz), axis=-1)


def quat_conj(q):
    return np.asarray(q, float) * np.array([-1.0, -1.0, -1.0, 1.0])


def quat_rotate(q, v):
    v = np.asarray(v, float)
    vq = np.concatenate((v, np.zeros(v.shape[:-1] + (1,))), axis=-1)
    return quat_mul(quat_mul(q, vq), quat_conj(q))[..., :3]


def quat_yaw(yaw):
    yaw = np.asarray(yaw, float)
    return np.stack((np.zeros_like(yaw), np.zeros_like(yaw), np.sin(yaw / 2), np.cos(yaw / 2)), axis=-1)


def quat_from_euler(roll, pitch, yaw):
    """xyzw from intrinsic Z-Y-X (yaw, pitch, roll), the convention of isaaclab.utils.math.quat_from_euler_xyz."""
    cr, sr = np.cos(np.asarray(roll) / 2), np.sin(np.asarray(roll) / 2)
    cp, sp = np.cos(np.asarray(pitch) / 2), np.sin(np.asarray(pitch) / 2)
    cy, sy = np.cos(np.asarray(yaw) / 2), np.sin(np.asarray(yaw) / 2)
    return np.stack((sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy,
                     cr * cp * cy + sr * sp * sy), axis=-1)


def yaw_of(q) -> float:
    x, y, z, w = (float(v) for v in q)
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


class PosedDuck:
    """One canonical asset instance whose rigid bodies follow world poses [B, 7] (xyzw) given per frame."""

    def __init__(self, stage, path: str, asset: str, body_names: list[str]):
        from pxr import Gf, Usd, UsdGeom, UsdPhysics

        prim = stage.DefinePrim(path, "Xform")
        prim.GetReferences().AddReference(asset)
        bodies = {}
        for p in Usd.PrimRange(prim):
            if p.HasAPI(UsdPhysics.RigidBodyAPI):
                bodies[p.GetName()] = p
                UsdPhysics.RigidBodyAPI(p).CreateRigidBodyEnabledAttr(False)
            if p.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI(p).CreateCollisionEnabledAttr(False)
            if p.IsA(UsdPhysics.Joint):
                UsdPhysics.Joint(p).CreateJointEnabledAttr(False)
            if p.HasAPI(UsdPhysics.ArticulationRootAPI):
                p.RemoveAPI(UsdPhysics.ArticulationRootAPI)
        self.names = list(body_names)
        self.parents = np.asarray([self.names.index(bodies[n].GetParent().GetName())
                                   if bodies[n].GetParent().GetName() in self.names else -1 for n in self.names])
        above = UsdGeom.XformCache().GetLocalToWorldTransform(bodies["trunk_base"].GetParent())
        if not Gf.IsClose(above, Gf.Matrix4d(1.0), 1e-9):
            raise RuntimeError("Expected identity above trunk_base")
        self.ops = []
        for name in self.names:
            xf = UsdGeom.Xformable(bodies[name])
            ops = {o.GetOpType(): o for o in xf.GetOrderedXformOps()}
            if UsdGeom.XformOp.TypeTranslate not in ops:
                xf.ClearXformOpOrder()
                ops = {UsdGeom.XformOp.TypeTranslate: xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble),
                       UsdGeom.XformOp.TypeOrient: xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)}
            scale = ops.get(UsdGeom.XformOp.TypeScale)
            if scale is not None and not Gf.IsClose(Gf.Vec3d(scale.Get()), Gf.Vec3d(1, 1, 1), 1e-6):
                raise RuntimeError(f"Non-unit scale on {name}")
            t, o = ops[UsdGeom.XformOp.TypeTranslate], ops[UsdGeom.XformOp.TypeOrient]
            self.ops.append((t, o, Gf.Vec3d if t.GetPrecision() == UsdGeom.XformOp.PrecisionDouble else Gf.Vec3f,
                             Gf.Quatd if o.GetPrecision() == UsdGeom.XformOp.PrecisionDouble else Gf.Quatf))
        self.prim = prim

    def local(self, pose: np.ndarray):
        pos, quat = pose[:, :3].astype(float), pose[:, 3:7].astype(float)
        has = self.parents >= 0
        parent = np.maximum(self.parents, 0)
        ppos = np.where(has[:, None], pos[parent], 0.0)
        pquat = np.where(has[:, None], quat[parent], np.array([0.0, 0.0, 0.0, 1.0]))
        return quat_rotate(quat_conj(pquat), pos - ppos), quat_mul(quat_conj(pquat), quat)

    def show(self, pose: np.ndarray, time=None) -> None:
        from pxr import Sdf, Usd

        code = Usd.TimeCode.Default() if time is None else Usd.TimeCode(time)
        lpos, lquat = self.local(pose)
        with Sdf.ChangeBlock():
            for b, (top, oop, V, Q) in enumerate(self.ops):
                p, q = lpos[b], lquat[b]
                top.Set(V(float(p[0]), float(p[1]), float(p[2])), code)
                oop.Set(Q(float(q[3]), float(q[0]), float(q[1]), float(q[2])), code)


class Grabber:
    """A UsdGeom camera with a Replicator RGB annotator. Lens values follow USD's tenths-of-a-unit convention."""

    def __init__(self, stage, path: str, width: int, height: int):
        import omni.replicator.core as rep
        from pxr import Gf, UsdGeom

        self.cam = UsdGeom.Camera.Define(stage, path)
        self.cam.CreateHorizontalApertureAttr(0.36)
        self.cam.CreateVerticalApertureAttr(0.36 * height / width)
        self.cam.CreateClippingRangeAttr(Gf.Vec2f(0.005, 30000.0))
        rp = rep.create.render_product(path, (width, height))
        self.ann = rep.AnnotatorRegistry.get_annotator("rgb")
        self.ann.attach([rp])

    def set(self, eye, target, focal_mm: float, fstop: float = 0.0, focus_m: float = 5.0) -> None:
        from dj import usdkit as K

        K.aim(self.cam.GetPrim(), eye, target)
        self.cam.GetFocalLengthAttr().Set(float(focal_mm) * 0.01)
        self.cam.GetFStopAttr().Set(float(fstop))
        self.cam.GetFocusDistanceAttr().Set(float(focus_m))

    def grab(self, updates: int, tries: int = 10):
        import omni.kit.app

        app = omni.kit.app.get_app()
        for i in range(updates + tries):
            app.update()
            if i + 1 < updates:
                continue
            d = self.ann.get_data()
            arr = np.asarray(d["data"] if isinstance(d, dict) else d)
            if arr.size:
                return arr[..., :3].copy()
        return None
