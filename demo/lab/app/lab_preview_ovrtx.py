#!/usr/bin/env python3
"""Render still previews of a Cascade Lab scene with OVRTX (no physics, no Isaac).

Authors the same stage the Isaac bridge builds before play: the shipped reBot USD as the root
layer, then demo/isaac_scene.prepare_scene() with the lab config. Cameras use the bridge's
cam0/side poses plus one wide hero view. Output: one PNG per camera and a contact sheet.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--source", required=True, help="Cascade Lab source root")
ap.add_argument("--config", required=True, help="scene config relative to source")
ap.add_argument("--out", required=True)
ap.add_argument("--width", type=int, default=1600)
ap.add_argument("--height", type=int, default=900)
ap.add_argument("--warmup", type=int, default=24)
args = ap.parse_args()

SRC = Path(args.source).resolve()
OUT = Path(args.out).resolve()
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(SRC / "demo"))

from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade  # noqa: E402

import isaac_scene  # noqa: E402
from own_kitchen import _box, _material  # noqa: E402

ROBOT = SRC / "assets/usd/RS-rebot-dev-arm/RS-rebot-dev-arm.usda"
CAMS = {
    "hero": ((1.35, -1.05, 0.72), (0.25, 0.05, 0.08), 20.0),
    "cam0": ((0.78, -0.35, 0.60), (0.28, 0.0, 0.0), 18.0),
    "side": ((0.95, 0.75, 0.45), (0.28, 0.0, 0.10), 18.0),
}


FACES = [[0, 1, 3, 2], [4, 6, 7, 5], [0, 4, 5, 1], [2, 3, 7, 6], [0, 2, 6, 4], [1, 5, 7, 3]]


def stub_cube(path, position, dimensions, color, dynamic=False, mass=None):
    """Mirror of scripts/isaac_bridge.py::_cube without physics simulation."""
    from pxr import UsdPhysics
    mesh = UsdGeom.Mesh.Define(STAGE, path)
    hx, hy, hz = [v / 2 for v in dimensions]
    mesh.CreatePointsAttr([Gf.Vec3f(sx * hx, sy * hy, sz * hz) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    mesh.CreateFaceVertexCountsAttr([4] * 6)
    mesh.CreateFaceVertexIndicesAttr([i for f in FACES for i in f])
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateExtentAttr([Gf.Vec3f(-hx, -hy, -hz), Gf.Vec3f(hx, hy, hz)])
    xf = UsdGeom.Xformable(mesh.GetPrim())
    xf.ClearXformOpOrder()
    xf.AddTransformOp().Set(Gf.Matrix4d(1).SetTranslate(Gf.Vec3d(*position)))
    mesh.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("boundingCube")
    if dynamic:
        UsdPhysics.RigidBodyAPI.Apply(mesh.GetPrim())
        UsdPhysics.MassAPI.Apply(mesh.GetPrim()).CreateMassAttr(mass or .05)
    return mesh


def look_at(eye, target):
    eye, target = np.array(eye, float), np.array(target, float)
    fwd = target - eye
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    m = Gf.Matrix4d(1)
    for i in range(3):
        m.SetRow(0, Gf.Vec4d(*right, 0))
        m.SetRow(1, Gf.Vec4d(*up, 0))
        m.SetRow(2, Gf.Vec4d(*(-fwd), 0))
        m.SetRow(3, Gf.Vec4d(*eye, 1))
    return m


t0 = time.time()
scene_path = OUT / "scene.usda"
STAGE = Usd.Stage.CreateNew(str(scene_path))
UsdGeom.SetStageUpAxis(STAGE, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(STAGE, 1.0)
STAGE.GetRootLayer().subLayerPaths.append(str(ROBOT))
config = isaac_scene.read_config(SRC / args.config)
specs = isaac_scene.prepare_scene(STAGE, config, cube=stub_cube, bind_pmat=lambda prim: None, base_z=0.0,
                                  is_playing=False)
for name, color, style in (("green_cube", (0.20, 0.62, 0.22), "stone"), ("pink_cube", (0.95, 0.42, 0.68), "puzzle")):
    pos = config["cube_positions"][name]
    dims = config["cube_dimensions"][name]
    mesh = stub_cube("/World_Props/" + name, pos, dims, color)
    try:
        isaac_scene.decorate_cube(STAGE, mesh, color, style=config.get("cube_surfaces", {}).get(name, style))
    except Exception as e:  # decoration is cosmetic in a preview
        print("cube decoration skipped:", name, type(e).__name__, e)
for cam, (eye, target, focal) in CAMS.items():
    c = UsdGeom.Camera.Define(STAGE, f"/LabCams/{cam}")
    c.CreateFocalLengthAttr(focal)
    c.CreateHorizontalApertureAttr(20.955)
    c.CreateVerticalApertureAttr(20.955 * args.height / args.width)
    c.CreateClippingRangeAttr(Gf.Vec2f(0.01, 50))
    UsdGeom.Xformable(c).AddTransformOp().Set(look_at(eye, target))
STAGE.GetRootLayer().Save()

products = []
layer = ["#usda 1.0", "", 'def Scope "LabRender"', "{"]
for cam in CAMS:
    layer += [f'    def RenderProduct "{cam}" (', '        prepend apiSchemas = ["OmniRtxSettingsCommonAdvancedAPI_1"]',
              "    ) {", f"        rel camera = [</LabCams/{cam}>]", "        uint[] deviceIds = [0]",
              '        token omni:rtx:background:source:type = "domeLight"',
              '        token omni:rtx:rendermode = "RealTimePathTracing"',
              "        bool omni:rtx:rt:ecoMode:enabled = false",
              '        token[] omni:rtx:waitForEvents = ["AllLoadingFinished", "OnlyOnFirstRequest"]',
              f"        rel orderedVars = [</LabRender/Vars/{cam}>]",
              f"        uniform int2 resolution = ({args.width}, {args.height})", "    }"]
    products.append(f"/LabRender/{cam}")
layer += ['    def "Vars"', "    {"]
for cam in CAMS:
    layer += [f'        def RenderVar "{cam}"', "        {", '            uniform string sourceName = "LdrColor"',
              "        }"]
layer += ["    }", "}"]
(OUT / "render_products.usda").write_text("\n".join(layer) + "\n")
STAGE.GetRootLayer().subLayerPaths.append("render_products.usda")
STAGE.GetRootLayer().Save()
print(f"stage authored in {time.time() - t0:.1f}s props={[s.name for s in specs]}", flush=True)

import ovrtx  # noqa: E402
from PIL import Image  # noqa: E402

renderer = ovrtx.Renderer(ovrtx.RendererConfig(log_file_path=str(OUT / "ovrtx.log"), log_level="warn"))
t1 = time.time()
renderer.open_usd(str(scene_path))
print(f"OVRTX {ovrtx.__version__} loaded in {time.time() - t1:.1f}s", flush=True)
result = None
for _ in range(args.warmup):
    result = renderer.step(render_products=set(products), delta_time=1 / 30)
images = {}
for prod in products:
    frame = result[prod].frames[0]
    name = prod.rsplit("/", 1)[1]
    key = f"/LabRender/Vars/{name}"
    var = frame.render_vars[key] if key in frame.render_vars else next(iter(frame.render_vars.values()))
    with var.map(device=ovrtx.Device.CPU) as mapping:
        img = np.array(np.from_dlpack(mapping))[..., :3]
    Image.fromarray(np.ascontiguousarray(img)).save(OUT / f"{name}.png")
    images[name] = img
    print("saved", name, img.shape, float(img.mean()), flush=True)
hero = Image.fromarray(images["hero"])
w, h = hero.size
sheet = Image.new("RGB", (w, h + h // 2), (8, 9, 10))
sheet.paste(hero, (0, 0))
for i, name in enumerate(("cam0", "side")):
    sheet.paste(Image.fromarray(images[name]).resize((w // 2, h // 2)), (i * w // 2, h))
sheet.save(OUT / "contact_sheet.jpg", quality=90)
(OUT / "summary.json").write_text(json.dumps({"config": args.config, "props": [s.name for s in specs],
                                               "identity": config.get("_identity"), "seconds": time.time() - t0},
                                              indent=1, default=str))
print("done", OUT / "contact_sheet.jpg", flush=True)
