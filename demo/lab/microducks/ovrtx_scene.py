"""Schoolyard OVRTX scene pieces shared by the offline renderer and the live server (USD authoring only)."""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "assets"))

import yard_layout as Y  # noqa: E402

SCOPE = "OvrtxFilm"


def log(m):
    print(f"[ducks-ovrtx {time.strftime('%H:%M:%S')}] {m}", flush=True)


def body_paths(duck):
    """USD paths of the duck's rigid bodies, in duck.names order."""
    from pxr import Usd, UsdPhysics
    found = {p.GetName(): str(p.GetPath()) for p in Usd.PrimRange(duck.prim) if p.HasAPI(UsdPhysics.RigidBodyAPI)}
    return [found[n] for n in duck.names]


def pose_to_rowmajor(pose):
    """[N, 7] pos + xyzw -> [N, 4, 4] USD (row-vector) world matrices."""
    p, q = pose[:, :3], pose[:, 3:7]
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
                  np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
                  np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], -2)
    M = np.zeros((len(pose), 4, 4))
    M[:, :3, :3] = np.transpose(R, (0, 2, 1))
    M[:, 3, :3] = p
    M[:, 3, 3] = 1.0
    return M


def oriented_box(stage, K, path, center, half, quat_xyzw, uv_per_m, mat):
    from pxr import Gf, UsdGeom
    m = K.boxmesh(stage, path, (0, 0, 0), tuple(2 * h for h in half), uv_per_m=uv_per_m)
    xf = UsdGeom.Xformable(m)
    xf.ClearXformOpOrder()
    xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*map(float, center)))
    x, y, z, w = quat_xyzw
    xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(float(w), float(x), float(y), float(z)))
    K.bind(m.GetPrim(), mat)
    return m


def quad(stage, K, path, corners, mat):
    """Textured quad, st 0..1 over the corners (bottom-left, bottom-right, top-right, top-left)."""
    from pxr import Gf, Sdf, UsdGeom, Vt
    m = UsdGeom.Mesh.Define(stage, path)
    m.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, c)) for c in corners]))
    m.CreateFaceVertexCountsAttr(Vt.IntArray([4]))
    m.CreateFaceVertexIndicesAttr(Vt.IntArray([0, 1, 2, 3]))
    pv = UsdGeom.PrimvarsAPI(m).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying)
    pv.Set(Vt.Vec2fArray([Gf.Vec2f(0, 0), Gf.Vec2f(1, 0), Gf.Vec2f(1, 1), Gf.Vec2f(0, 1)]))
    m.CreateDoubleSidedAttr(True)
    m.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    K.bind(m.GetPrim(), mat)
    return m


def build_yard(stage, tex, sky, dome_intensity=520.0, sun_intensity=3000.0):
    from pxr import Gf, UsdGeom, UsdLux
    from usdkit import (bind, boxmesh, cylinder, frustum, surface)
    import usdkit as K
    T = lambda n: str(Path(tex) / n)  # noqa: E731
    M = {
        "yard": surface(stage, "/World/Looks/Yard", albedo_tex=T("yard_floor.png"), roughness=0.85, wrap="clamp"),
        "grass": surface(stage, "/World/Looks/Grass", albedo_tex=T("grass.png"), roughness=0.95, wrap="repeat"),
        "brick": surface(stage, "/World/Looks/Brick", albedo_tex=T("brick.png"), roughness=0.85, wrap="repeat"),
        "coping": surface(stage, "/World/Looks/Coping", color=(0.80, 0.78, 0.74), roughness=0.7),
        "facade": surface(stage, "/World/Looks/Facade", albedo_tex=T("facade.png"), roughness=0.8, wrap="clamp"),
        "facadeWall": surface(stage, "/World/Looks/FacadeWall", color=(0.86, 0.78, 0.62), roughness=0.85),
        "roof": surface(stage, "/World/Looks/Roof", color=(0.50, 0.20, 0.16), roughness=0.7),
        "mural": surface(stage, "/World/Looks/Mural", albedo_tex=T("mural.png"), roughness=0.8, wrap="clamp"),
        "ramp": surface(stage, "/World/Looks/Ramp", albedo_tex=T("ramp_rubber.png"), roughness=0.9, wrap="repeat"),
        "rampTop": surface(stage, "/World/Looks/RampTop", color=(0.20, 0.48, 0.80), roughness=0.75),
        "stripe": surface(stage, "/World/Looks/Stripe", color=(0.95, 0.78, 0.10), roughness=0.6),
        "wood": surface(stage, "/World/Looks/Wood", albedo_tex=T("wood.png"), roughness=0.7, wrap="repeat"),
        "cone": surface(stage, "/World/Looks/Cone", color=(0.95, 0.35, 0.05), roughness=0.45),
        "coneW": surface(stage, "/World/Looks/ConeW", color=(0.92, 0.92, 0.92), roughness=0.45),
        "bush": surface(stage, "/World/Looks/Bush", color=(0.16, 0.42, 0.14), roughness=0.9),
        "bark": surface(stage, "/World/Looks/Bark", color=(0.36, 0.25, 0.16), roughness=0.95),
        "leaves": surface(stage, "/World/Looks/Leaves", color=(0.20, 0.45, 0.16), roughness=0.9),
        "leaves2": surface(stage, "/World/Looks/Leaves2", color=(0.28, 0.52, 0.20), roughness=0.9),
        "frame": surface(stage, "/World/Looks/SwingFrame", color=(0.16, 0.38, 0.80), roughness=0.4, metallic=0.3),
        "seat": surface(stage, "/World/Looks/Seat", color=(0.85, 0.15, 0.12), roughness=0.5),
    }
    floor = boxmesh(stage, "/World/Yard/floor", (0, 0, -0.01), (2 * Y.YARD, 2 * Y.YARD, 0.02), uv_per_m=1 / (2 * Y.YARD))
    bind(floor.GetPrim(), M["yard"])
    grass = boxmesh(stage, "/World/Yard/grass", (0, 0, -0.03), (30.0, 30.0, 0.02), uv_per_m=0.7)
    bind(grass.GetPrim(), M["grass"])
    for name, c, hs in Y.wall_boxes():
        oriented_box(stage, K, f"/World/Yard/{name}", c, hs, (0, 0, 0, 1), 1.6, M["brick"])
        oriented_box(stage, K, f"/World/Yard/{name}_cap", (c[0], c[1], Y.WALL_H + 0.008),
                     (hs[0] + 0.006, hs[1] + 0.006, 0.008), (0, 0, 0, 1), 1.0, M["coping"])
    fy = Y.YARD + 0.45
    oriented_box(stage, K, "/World/Yard/facade_block", (0, fy, 0.45), (2.9, 0.25, 0.45), (0, 0, 0, 1), 1.0,
                 M["facadeWall"])
    oriented_box(stage, K, "/World/Yard/roof", (0, fy, 0.93), (3.0, 0.34, 0.04), (0, 0, 0, 1), 1.0, M["roof"])
    quad(stage, K, "/World/Yard/facade_art", [(-2.9, fy - 0.252, 0), (2.9, fy - 0.252, 0), (2.9, fy - 0.252, 0.9),
                                              (-2.9, fy - 0.252, 0.9)], M["facade"])
    mx = -Y.YARD - 0.02
    oriented_box(stage, K, "/World/Yard/mural_panel", (mx - 0.012, -0.6, 2 * 0.07 + 0.14), (0.012, 0.92, 0.13),
                 (0, 0, 0, 1), 1.0, M["coping"])
    quad(stage, K, "/World/Yard/mural_art", [(mx + 0.001, -1.5, 0.16), (mx + 0.001, 0.3, 0.16), (mx + 0.001, 0.3, 0.40),
                                             (mx + 0.001, -1.5, 0.40)], M["mural"])
    k = 0
    for r in Y.RAMPS:
        for name, c, hs, q, role in Y.ramp_boxes(r):
            oriented_box(stage, K, f"/World/Yard/{name}", c, hs, q, 4.0, M["rampTop" if role == "top" else "ramp"])
            k += 1
    for i, (x, y) in enumerate(Y.CONES):
        bind(frustum(stage, f"/World/Yard/cone{i}_base", 0.05, 0.05, 0.0, 0.008, seg=4, center=(x, y, 0)).GetPrim(),
             M["cone"])
        for j, (z0, z1, m) in enumerate([(0.008, 0.06, "cone"), (0.06, 0.09, "coneW"), (0.09, 0.14, "cone")]):
            r0 = Y.CONE_R * (1 - z0 / 0.16)
            r1 = Y.CONE_R * (1 - z1 / 0.16)
            bind(frustum(stage, f"/World/Yard/cone{i}_{j}", r0, r1, z0, z1, seg=48, center=(x, y, 0)).GetPrim(), M[m])
    for i, (x, y, yaw) in enumerate(Y.PLANTERS):
        hs = tuple(v / 2 for v in Y.PLANTER_SIZE)
        oriented_box(stage, K, f"/World/Yard/planter{i}", (x, y, hs[2]), hs, Y.quat_xyzw_from_yaw_pitch(yaw), 3.0,
                     M["wood"])
        s = UsdGeom.Sphere.Define(stage, f"/World/Yard/planter{i}_bush")
        s.CreateRadiusAttr(1.0)
        xf = UsdGeom.Xformable(s)
        xf.AddTranslateOp().Set(Gf.Vec3d(x, y, 0.095))
        xf.AddRotateZOp().Set(math.degrees(yaw))
        xf.AddScaleOp().Set(Gf.Vec3f(0.2, 0.07, 0.055))
        bind(s.GetPrim(), M["bush"])
    for i, (x, y, sc) in enumerate([(-3.1, 3.0, 1.0), (3.2, 2.9, 1.2), (-3.3, -2.6, 0.9), (3.0, -3.2, 1.1),
                                    (-0.9, 3.6, 0.8)]):
        bind(cylinder(stage, f"/World/Trees/t{i}_trunk", 0.05 * sc, 0.7 * sc, center=(x, y, 0.35 * sc)).GetPrim(),
             M["bark"])
        for j, (dx, dy, dz, r) in enumerate([(0, 0, 0.85, 0.38), (0.2, 0.1, 0.75, 0.28), (-0.18, -0.12, 0.78, 0.3),
                                             (0.05, -0.2, 1.05, 0.25)]):
            s = UsdGeom.Sphere.Define(stage, f"/World/Trees/t{i}_c{j}")
            s.CreateRadiusAttr(r * sc)
            UsdGeom.Xformable(s).AddTranslateOp().Set(Gf.Vec3d(x + dx * sc, y + dy * sc, dz * sc))
            bind(s.GetPrim(), M["leaves" if j % 2 else "leaves2"])
    sx, sy = 3.2, 0.0
    j = 0
    for a, b in ([((sx - 0.25, sy + dy, 0), (sx, sy + dy, 0.75)) for dy in (-0.4, 0.4)] +
                 [((sx + 0.25, sy + dy, 0), (sx, sy + dy, 0.75)) for dy in (-0.4, 0.4)] +
                 [((sx, sy - 0.42, 0.75), (sx, sy + 0.42, 0.75))]):
        a, b = np.array(a), np.array(b)
        c = UsdGeom.Capsule.Define(stage, f"/World/Swing/bar{j}")
        c.CreateRadiusAttr(0.018)
        c.CreateHeightAttr(float(np.linalg.norm(b - a)))
        c.CreateAxisAttr("Z")
        K.aim(c.GetPrim(), (a + b) / 2, b + (b - a) * 10)
        bind(c.GetPrim(), M["frame"])
        j += 1
    for dy in (-0.12, 0.12):
        oriented_box(stage, K, f"/World/Swing/seat{j}", (sx, sy + dy, 0.27), (0.04, 0.1, 0.008), (0, 0, 0, 1), 1.0,
                     M["seat"])
        j += 1
    # light: HDR sky dome if present, plus a sun
    dome = UsdLux.DomeLight.Define(stage, "/World/Sky/Dome")
    if sky and Path(sky).exists():
        dome.CreateTextureFileAttr(str(sky))
        dome.CreateTextureFormatAttr("latlong")
        dome.CreateIntensityAttr(float(dome_intensity))
    else:
        dome.CreateIntensityAttr(700.0)
        dome.CreateColorAttr((0.62, 0.76, 0.98))
    sun = UsdLux.DistantLight.Define(stage, "/World/Sky/Sun")
    sun.CreateIntensityAttr(float(sun_intensity))
    sun.CreateAngleAttr(0.8)
    sun.CreateColorAttr((1.0, 0.95, 0.86))
    K.aim(sun.GetPrim(), (2.0, -3.0, 6.0), (0.0, 0.0, 0.0))


def color_heads(stage, ducks, only=None):
    import os
    """Per-duck head colour: de-instance the head shell parts only and bind a colour on their meshes."""
    from pxr import Usd, UsdGeom, UsdShade
    from usdkit import surface
    for i, d in enumerate(ducks):
        if only is not None and i not in (only if isinstance(only, (set, list, tuple)) else {only}):
            continue
        c = Y.DUCK_COLORS[i]
        for item in os.environ.get("HOLO_HEAD_COLORS", "").split(";"):  # "10=0.95,0.25,0.25;6=..." (1-based ids)
            if item.startswith(f"{i + 1}="):
                c = tuple(float(v) for v in item.split("=")[1].split(","))
        mat = surface(stage, f"/World/Looks/DuckHead{i:02d}", color=c, roughness=0.35)
        shells = [p for p in Usd.PrimRange(d.prim) if "head_shell" in p.GetName() and p.IsInstance()]
        for p in shells:
            p.SetInstanceable(False)
        n = 0
        for p in shells:
            for q in Usd.PrimRange(p):
                if q.IsA(UsdGeom.Mesh):
                    UsdShade.MaterialBindingAPI.Apply(q).Bind(mat, UsdShade.Tokens.strongerThanDescendants)
                    n += 1
        if i == 0:
            log(f"head shells recoloured per duck: {len(shells)} parts, {n} meshes on duck 1")


def head_camera(stage, duck, path):
    """Camera rigidly on jaw_soft, same mount as the robot's head camera (looks forward, up = +z at rest)."""
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    jaw = next(p for p in Usd.PrimRange(duck.prim) if p.GetName() == "jaw_soft" and p.HasAPI(UsdPhysics.RigidBodyAPI))
    cam = UsdGeom.Camera.Define(stage, str(jaw.GetPath().AppendChild(path)))
    cam.CreateHorizontalApertureAttr(0.36)
    cam.CreateVerticalApertureAttr(0.36 * 9 / 16)
    cam.CreateFocalLengthAttr(0.132)  # 75 deg vertical field of view, as the sim head camera
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.004, 1000.0))
    xf = UsdGeom.Xformable(cam)
    xf.ClearXformOpOrder()
    xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(0.0155, 0.0, -0.0853))
    xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(0.70710678, 0.0, 0.0, -0.70710678))
    return str(cam.GetPath())


def product_layer(products, mode="RealTimePathTracing", device=0):
    """products: list of (name, camera path, w, h). BG_TEXTURE/BG_SCALE/BG_MODE env: visible background texture
    (separate from the dome light that lights the scene)."""
    import os
    bgt = os.environ.get("BG_TEXTURE")
    if bgt:
        bg = (f'token omni:rtx:background:source:type = "texture"\n'
              f'        asset omni:rtx:background:source:texture:path = @{bgt}@\n'
              f'        float omni:rtx:background:source:texture:luminanceScale = {float(os.environ.get("BG_SCALE", "1.0"))}')
        if os.environ.get("BG_MODE"):
            bg += f'\n        token omni:rtx:background:source:texture:textureMode = "{os.environ["BG_MODE"]}"'
    else:
        bg = 'token omni:rtx:background:source:type = "domeLight"'
    body = []
    for name, campath, w, h in products:
        body.append(f'''
    def RenderProduct "{name}" (
        prepend apiSchemas = ["OmniRtxSettingsCommonAdvancedAPI_1"]
    ) {{
        rel camera = [<{campath}>]
        uint[] deviceIds = [{device}]
        {bg}
        token omni:rtx:rendermode = "{mode}"
        bool omni:rtx:rt:ecoMode:enabled = false
        token[] omni:rtx:waitForEvents = ["AllLoadingFinished", "OnlyOnFirstRequest"]
        rel orderedVars = [</{SCOPE}/Vars/LdrColor>]
        uniform int2 resolution = ({w}, {h})
    }}''')
    return f'''#usda 1.0

def Scope "{SCOPE}"
{{{"".join(body)}

    def "Vars"
    {{
        def RenderVar "LdrColor"
        {{
            uniform string sourceName = "LdrColor"
        }}
    }}
}}
'''




# ============================================================================= look-dev v2 (premium)
def pbr(stage, path, diff=None, rough=None, nor=None, color=(0.5, 0.5, 0.5), roughness=0.6, metallic=0.0,
        tint=None, wrap="repeat", nor_strength=1.0, clearcoat=None, clearcoat_roughness=0.03, emissive=None,
        emissive_tex=None, emissive_scale=1.0):
    """UsdPreviewSurface with optional albedo / roughness / OpenGL normal maps on the st primvar."""
    from pxr import Gf, Sdf, UsdShade
    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, f"{path}/Surface")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(float(metallic))
    sh.CreateInput("useSpecularWorkflow", Sdf.ValueTypeNames.Int).Set(0)
    if clearcoat is not None:
        sh.CreateInput("clearcoat", Sdf.ValueTypeNames.Float).Set(float(clearcoat))
        sh.CreateInput("clearcoatRoughness", Sdf.ValueTypeNames.Float).Set(float(clearcoat_roughness))
    rd = None
    if diff or rough or nor or emissive_tex:
        rd = UsdShade.Shader.Define(stage, f"{path}/st")
        rd.CreateIdAttr("UsdPrimvarReader_float2")
        rd.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")

    def tex(name, f, cs, out, outtype, scale=None, bias=None):
        t = UsdShade.Shader.Define(stage, f"{path}/{name}")
        t.CreateIdAttr("UsdUVTexture")
        t.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(str(f))
        t.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set(cs)
        t.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set(wrap)
        t.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set(wrap)
        if scale is not None:
            t.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(*scale))
        if bias is not None:
            t.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(*bias))
        t.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(rd.ConnectableAPI(), "result")
        t.CreateOutput(out, outtype)
        return t

    if diff:
        t = tex("Diff", diff, "sRGB", "rgb", Sdf.ValueTypeNames.Float3, scale=(*(tint or (1, 1, 1)), 1.0))
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(t.ConnectableAPI(), "rgb")
    else:
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    if rough:
        t = tex("Rough", rough, "raw", "r", Sdf.ValueTypeNames.Float)
        sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).ConnectToSource(t.ConnectableAPI(), "r")
    else:
        sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(roughness))
    if emissive_tex:
        e = float(emissive_scale)
        t = tex("Emis", emissive_tex, "sRGB", "rgb", Sdf.ValueTypeNames.Float3, scale=(e, e, e, 1.0))
        sh.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(t.ConnectableAPI(), "rgb")
    elif emissive is not None:
        sh.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*emissive))
    if nor:
        k = float(nor_strength)
        t = tex("Nor", nor, "raw", "rgb", Sdf.ValueTypeNames.Float3, scale=(2 * k, 2 * k, 2, 1), bias=(-k, -k, -1, 0))
        sh.CreateInput("normal", Sdf.ValueTypeNames.Normal3f).ConnectToSource(t.ConnectableAPI(), "rgb")
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return mat


def fit_asset(stage, path, usd, center_xy, yaw, size_xyz=None, height=None, z0=0.0):
    """Reference a USD asset and scale it uniformly to fit a footprint (or a height), base at z0."""
    from pxr import Gf, Usd, UsdGeom
    src = Usd.Stage.Open(str(usd))
    up = UsdGeom.GetStageUpAxis(src)
    mpu = UsdGeom.GetStageMetersPerUnit(src) or 1.0
    root = stage.DefinePrim(path, "Xform")
    inner = stage.DefinePrim(path + "/asset", "Xform")
    inner.GetReferences().AddReference(str(usd))
    ix = UsdGeom.Xformable(inner)
    ix.ClearXformOpOrder()
    if up == "Y":
        ix.AddRotateXOp().Set(90.0)
    bb = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"]).ComputeWorldBound(inner).ComputeAlignedRange()
    lo, hi = np.array(bb.GetMin()), np.array(bb.GetMax())
    ext = hi - lo
    if height is not None:
        s = height / max(ext[2], 1e-6)
    else:
        s = min(size_xyz[0] / max(ext[0], 1e-6), size_xyz[1] / max(ext[1], 1e-6), size_xyz[2] / max(ext[2], 1e-6))
    c = (lo + hi) / 2
    rx = UsdGeom.Xformable(root)
    rx.ClearXformOpOrder()
    rx.AddTranslateOp().Set(Gf.Vec3d(float(center_xy[0]), float(center_xy[1]), float(z0)))
    rx.AddRotateZOp().Set(float(math.degrees(yaw)))
    rx.AddScaleOp().Set(Gf.Vec3f(float(s), float(s), float(s)))
    rx.AddTranslateOp(opSuffix="center").Set(Gf.Vec3d(float(-c[0]), float(-c[1]), float(-lo[2])))
    del mpu
    return root


def build_yard_v2(stage, tex, sky, ph, dome_intensity=900.0, sun_intensity=3400.0, v3=False, sun_from=(2.5, -3.5, 5.0),
                  sun_color=(1.0, 0.95, 0.86)):
    from pxr import Gf, UsdGeom, UsdLux
    import usdkit as K
    from usdkit import bind, boxmesh, frustum
    tex, ph = Path(tex), Path(ph)
    T = lambda n: str(tex / n)  # noqa: E731
    PH = lambda a, n: str(ph / a / n)  # noqa: E731
    M = {
        "yard": pbr(stage, "/World/Looks/YardAsphalt", T("yard_v3_diff.jpg" if v3 else "yard_v2_diff.jpg"), T("yard_v2_rough.jpg"),
                    T("yard_v2_nor.jpg"), wrap="clamp", nor_strength=0.8),
        "grass": pbr(stage, "/World/Looks/Grass", T("grass_v3.png" if v3 else "grass.png"), roughness=0.95),
        "concrete": pbr(stage, "/World/Looks/Concrete", PH("concrete_wall_001", "concrete_wall_001_diff_2k.jpg"),
                        PH("concrete_wall_001", "concrete_wall_001_rough_2k.jpg"),
                        PH("concrete_wall_001", "concrete_wall_001_nor_gl_2k.jpg"),
                        tint=(0.82, 0.82, 0.8) if v3 else (0.95, 0.93, 0.9), nor_strength=1.4 if v3 else 1.0),
        "coping": pbr(stage, "/World/Looks/Coping", PH("concrete_wall_001", "concrete_wall_001_diff_2k.jpg"),
                      PH("concrete_wall_001", "concrete_wall_001_rough_2k.jpg"), tint=(1.15, 1.13, 1.1)),
        "plaster": (pbr(stage, "/World/Looks/Plaster", T("plaster_v3_diff.jpg"),
                        PH("painted_plaster_wall", "painted_plaster_wall_rough_2k.jpg"),
                        PH("painted_plaster_wall", "painted_plaster_wall_nor_gl_2k.jpg"), nor_strength=1.3) if v3 else
                    pbr(stage, "/World/Looks/Plaster", PH("painted_plaster_wall", "painted_plaster_wall_diff_2k.jpg"),
                        PH("painted_plaster_wall", "painted_plaster_wall_rough_2k.jpg"),
                        PH("painted_plaster_wall", "painted_plaster_wall_nor_gl_2k.jpg"), tint=(1.0, 0.9, 0.72))),
        "plinth": pbr(stage, "/World/Looks/Plinth", PH("concrete_wall_001", "concrete_wall_001_diff_2k.jpg"),
                      PH("concrete_wall_001", "concrete_wall_001_rough_2k.jpg"), tint=(0.62, 0.55, 0.5)),
        "roof": pbr(stage, "/World/Looks/Roof", color=(0.42, 0.16, 0.12), roughness=0.55),
        "frame": pbr(stage, "/World/Looks/WinFrame", color=(0.58, 0.58, 0.56) if v3 else (0.93, 0.93, 0.9),
                     roughness=0.45 if v3 else 0.35),
        "glass": pbr(stage, "/World/Looks/Glass", color=(0.10, 0.16, 0.22), roughness=0.04, metallic=0.6),
        "door": pbr(stage, "/World/Looks/Door", color=(0.10, 0.32, 0.50), roughness=0.4),
        "sign": pbr(stage, "/World/Looks/Sign", T("sign_v2.png"), roughness=0.5, wrap="clamp"),
        "rubberR": pbr(stage, "/World/Looks/RubberRed", PH("rubber_tiles", "rubber_tiles_diff_2k.jpg"),
                       PH("rubber_tiles", "rubber_tiles_rough_2k.jpg"), PH("rubber_tiles", "rubber_tiles_nor_gl_2k.jpg"),
                       tint=(8.0, 2.5, 2.0) if v3 else (3.2, 1.0, 0.75)),
        "rubberB": pbr(stage, "/World/Looks/RubberBlue", PH("rubber_tiles", "rubber_tiles_diff_2k.jpg"),
                       PH("rubber_tiles", "rubber_tiles_rough_2k.jpg"), PH("rubber_tiles", "rubber_tiles_nor_gl_2k.jpg"),
                       tint=(2.0, 3.6, 6.5) if v3 else (0.75, 1.6, 3.2)),
        "cone": pbr(stage, "/World/Looks/Cone", color=(0.95, 0.33, 0.05), roughness=0.4),
        "coneW": pbr(stage, "/World/Looks/ConeW", color=(0.92, 0.92, 0.92), roughness=0.4),
        "seat": pbr(stage, "/World/Looks/Seat", color=(0.85, 0.15, 0.12), roughness=0.5),
        "steel": pbr(stage, "/World/Looks/Steel", color=(0.16, 0.38, 0.80), roughness=0.35, metallic=0.4),
    }
    floor = boxmesh(stage, "/World/Yard/floor", (0, 0, -0.01), (2 * Y.YARD, 2 * Y.YARD, 0.02),
                    uv_per_m=1 / (2 * Y.YARD))
    bind(floor.GetPrim(), M["yard"])
    bind(boxmesh(stage, "/World/Yard/grass", (0, 0, -0.03), (40.0, 40.0, 0.02), uv_per_m=0.7).GetPrim(), M["grass"])
    for name, c, hs in Y.wall_boxes():
        oriented_box(stage, K, f"/World/Yard/{name}", c, hs, (0, 0, 0, 1), 2.5, M["concrete"])
        oriented_box(stage, K, f"/World/Yard/{name}_cap", (c[0], c[1], Y.WALL_H + 0.008),
                     (hs[0] + 0.008, hs[1] + 0.008, 0.008), (0, 0, 0, 1), 2.5, M["coping"])
    # school building (behind the north wall): plaster block, plinth, windows, door, sign, roof
    fy, fd, fh, fw = Y.YARD + 0.45, 0.25, 0.9, 2.9
    front = fy - fd
    oriented_box(stage, K, "/World/School/block", (0, fy, fh / 2), (fw, fd, fh / 2), (0, 0, 0, 1), 1.5, M["plaster"])
    oriented_box(stage, K, "/World/School/plinth", (0, front - 0.004, 0.05), (fw + 0.01, 0.006, 0.05), (0, 0, 0, 1),
                 2.5, M["plinth"])
    oriented_box(stage, K, "/World/School/roof", (0, fy, fh + 0.035), (fw + 0.1, fd + 0.09, 0.035), (0, 0, 0, 1), 1.0,
                 M["roof"])
    for row, z in enumerate((0.25, 0.55)):
        for i in range(10):
            x = -2.6 + i * 0.578
            if row == 0 and i in (4, 5):
                continue
            oriented_box(stage, K, f"/World/School/win{row}_{i}_f", (x, front - 0.006, z), (0.105, 0.006, 0.085),
                         (0, 0, 0, 1), 1.0, M["frame"])
            oriented_box(stage, K, f"/World/School/win{row}_{i}_g", (x, front - 0.0125, z), (0.09, 0.002, 0.07),
                         (0, 0, 0, 1), 1.0, M["glass"])
            oriented_box(stage, K, f"/World/School/win{row}_{i}_m", (x, front - 0.0145, z), (0.004, 0.001, 0.07),
                         (0, 0, 0, 1), 1.0, M["frame"])
    oriented_box(stage, K, "/World/School/door_f", (0.0, front - 0.006, 0.17), (0.2, 0.006, 0.17), (0, 0, 0, 1), 1.0,
                 M["frame"])
    oriented_box(stage, K, "/World/School/door", (0.0, front - 0.012, 0.16), (0.18, 0.003, 0.16), (0, 0, 0, 1), 1.0,
                 M["door"])
    oriented_box(stage, K, "/World/School/sign_board", (0.0, front - 0.01, 0.775), (0.62, 0.01, 0.0775), (0, 0, 0, 1),
                 1.0, M["frame"])
    quad(stage, K, "/World/School/sign_art", [(-0.6, front - 0.0205, 0.7), (0.6, front - 0.0205, 0.7),
                                              (0.6, front - 0.0205, 0.85), (-0.6, front - 0.0205, 0.85)], M["sign"])
    # ramps: playground rubber, red slopes and blue tops
    for r in Y.RAMPS:
        for name, c, hs, q, role in Y.ramp_boxes(r):
            oriented_box(stage, K, f"/World/Yard/{name}", c, hs, q, 3.0, M["rubberB" if role == "top" else "rubberR"])
    for i, (x, y) in enumerate(Y.CONES):
        bind(frustum(stage, f"/World/Yard/cone{i}_base", 0.05, 0.05, 0.0, 0.008, seg=4, center=(x, y, 0)).GetPrim(),
             M["cone"])
        for j, (za, zb, m) in enumerate([(0.008, 0.06, "cone"), (0.06, 0.09, "coneW"), (0.09, 0.14, "cone")]):
            bind(frustum(stage, f"/World/Yard/cone{i}_{j}", Y.CONE_R * (1 - za / 0.16), Y.CONE_R * (1 - zb / 0.16),
                         za, zb, seg=48, center=(x, y, 0)).GetPrim(), M[m])
    for i, (x, y, yaw) in enumerate(Y.PLANTERS):
        fit_asset(stage, f"/World/Yard/planter{i}", ph / "planter_box_01" / "planter_box_01_2k.usdc", (x, y), yaw,
                  size_xyz=Y.PLANTER_SIZE)
        fit_asset(stage, f"/World/Yard/planter{i}_shrub", ph / "shrub_02" / "shrub_02_2k.usdc", (x, y), yaw,
                  size_xyz=(Y.PLANTER_SIZE[0] * 0.9, Y.PLANTER_SIZE[1] * 1.3, 0.16), z0=Y.PLANTER_SIZE[2] * 0.75)
    for i, (x, y, h, yaw) in enumerate([(-3.1, 3.0, 1.5, 0.3), (3.2, 2.9, 1.8, 1.4), (-3.3, -2.6, 1.3, 2.2),
                                        (3.0, -3.2, 1.6, 4.0), (-1.2, 3.7, 1.2, 5.1), (1.6, 3.8, 1.4, 0.9)]):
        fit_asset(stage, f"/World/Trees/t{i}", ph / "tree_small_02" / "tree_small_02_2k.usdc", (x, y), yaw, height=h)
    sx, sy = 3.2, 0.0
    j = 0
    for a, b in ([((sx - 0.25, sy + dy, 0), (sx, sy + dy, 0.75)) for dy in (-0.4, 0.4)] +
                 [((sx + 0.25, sy + dy, 0), (sx, sy + dy, 0.75)) for dy in (-0.4, 0.4)] +
                 [((sx, sy - 0.42, 0.75), (sx, sy + 0.42, 0.75))]):
        a, b = np.array(a), np.array(b)
        c = UsdGeom.Capsule.Define(stage, f"/World/Swing/bar{j}")
        c.CreateRadiusAttr(0.018)
        c.CreateHeightAttr(float(np.linalg.norm(b - a)))
        c.CreateAxisAttr("Z")
        K.aim(c.GetPrim(), (a + b) / 2, b + (b - a) * 10)
        bind(c.GetPrim(), M["steel"])
        j += 1
    for dy in (-0.12, 0.12):
        oriented_box(stage, K, f"/World/Swing/seat{j}", (sx, sy + dy, 0.27), (0.04, 0.1, 0.008), (0, 0, 0, 1), 1.0,
                     M["seat"])
        j += 1
    dome = UsdLux.DomeLight.Define(stage, "/World/Sky/Dome")
    if sky and Path(sky).exists():
        dome.CreateTextureFileAttr(str(sky))
        dome.CreateTextureFormatAttr("latlong")
    dome.CreateIntensityAttr(float(dome_intensity))
    sun = UsdLux.DistantLight.Define(stage, "/World/Sky/Sun")
    sun.CreateIntensityAttr(float(sun_intensity))
    sun.CreateAngleAttr(1.2)
    sun.CreateColorAttr(tuple(sun_color))
    K.aim(sun.GetPrim(), tuple(sun_from), (0.0, 0.0, 0.0))


def duck_rings(stage, n):
    """Team Holodeck-dark placement rings (render only): outer 0.150-0.162 m, inner 0.095-0.100 m, emissive green.
    Returns one translate op per duck, to be set per frame from the base pose."""
    from pxr import Gf, UsdGeom
    from usdkit import bind, tube
    import os
    k = float(os.environ.get("HOLO_RING_EMIS", "1.0"))
    mat = pbr(stage, "/World/Looks/DuckRing", color=(0.05, 0.2, 0.02), roughness=0.4, emissive=(0.85 * k, 2.19 * k, 0.05 * k))
    ops = []
    for i in range(n):
        xf = UsdGeom.Xform.Define(stage, f"/World/Rings/Ring{i:02d}")
        ops.append(xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble))
        for nm, r0, r1 in (("outer", 0.150, 0.162), ("inner", 0.095, 0.100)):
            bind(tube(stage, f"/World/Rings/Ring{i:02d}/{nm}", r0, r1, 0.0012, 0.0016, seg=96).GetPrim(), mat)
    return ops


def build_holodeck(stage, ph, grid_tex, dome_intensity=650.0, key_intensity=5200.0, fill_intensity=900.0,
                   rim_intensity=700.0, key_from=(-2.2, -0.7, 3.0), wall_tint=(0.95, 1.03, 1.10)):
    """Team Holodeck light look (storyboard_concept/holodeck_light.usda: grid floor, roughness 0.16, clearcoat 0.7,
    cool dome) rebuilt around the yard's physical layout. Every collider of the simulation is rendered with its real
    size (ramps, cones as gaffer-tape stacks, planters as sandbags); team-style props stand outside the walking area."""
    from pxr import Gf, UsdGeom, UsdLux
    import usdkit as K
    from usdkit import bind, boxmesh, frustum, tube
    ph = Path(ph)
    PH = lambda a, n: str(ph / a / f"{a}_{n}_2k.jpg")  # noqa: E731

    def ph_mat(path, aid, tint=None, roughness=None, metallic=0.0, nor_strength=1.0):
        return pbr(stage, path, PH(aid, "diff"), PH(aid, "rough") if roughness is None else None, PH(aid, "nor_gl"),
                   tint=tint, roughness=roughness or 0.6, metallic=metallic, nor_strength=nor_strength)
    import os
    floor_tex, wall_col = os.environ.get("HOLO_FLOOR_TEX"), os.environ.get("HOLO_WALL_COLOR")
    M = {
        "grid": (pbr(stage, "/World/Looks/HoloDarkFloor", color=(0.006, 0.006, 0.007),
                     roughness=float(os.environ.get("HOLO_DARK_ROUGH", "0.07")), clearcoat=1.0,
                     clearcoat_roughness=float(os.environ.get("HOLO_DARK_ROUGH", "0.02")), emissive_tex=str(grid_tex),
                     emissive_scale=float(os.environ.get("HOLO_GRID_EMIS", "1.0"))) if os.environ.get("HOLO_DARK") else
                 pbr(stage, "/World/Looks/BackstageConcrete", floor_tex, roughness=float(os.environ.get("HOLO_FLOOR_ROUGH", "0.3")),
                     clearcoat=0.45, clearcoat_roughness=0.2, tint=None) if floor_tex else
                 pbr(stage, "/World/Looks/HoloGrid", str(grid_tex), roughness=0.38, clearcoat=0.35, clearcoat_roughness=0.25)),
        "wall": (pbr(stage, "/World/Looks/BackstageWall", color=tuple(float(x) for x in wall_col.split(",")), roughness=0.85)
                 if wall_col else pbr(stage, "/World/Looks/HoloWall", str(grid_tex), roughness=0.5, tint=wall_tint)),
        "plywood": ph_mat("/World/Looks/Plywood", "plywood"),
        "carpet": pbr(stage, "/World/Looks/Carpet", None, None, PH("dirty_carpet", "nor_gl"), color=(0.30, 0.36, 0.46),
                      roughness=0.95, nor_strength=1.6),
        "metal": pbr(stage, "/World/Looks/MetalPlate", None, None, PH("metal_plate", "nor_gl"), color=(0.63, 0.65, 0.68),
                     roughness=0.38, metallic=0.9, nor_strength=1.2),
        "rubber": ph_mat("/World/Looks/RubberMat", "rubberized_track", tint=(0.42, 0.44, 0.46), nor_strength=0.8),
        "tape": pbr(stage, "/World/Looks/GaffTape", color=(0.035, 0.035, 0.04), roughness=0.55),
        "tape_silver": pbr(stage, "/World/Looks/GaffTapeSilver", color=(0.55, 0.56, 0.58), roughness=0.4, metallic=0.5),
        "spike": pbr(stage, "/World/Looks/SpikeTape", color=(0.20, 0.21, 0.22), roughness=0.6),
        "tape_core": pbr(stage, "/World/Looks/TapeCore", color=(0.55, 0.47, 0.36), roughness=0.8),
        "canvas": pbr(stage, "/World/Looks/Canvas", PH("dirty_carpet", "diff"), roughness=0.9, tint=(0.09, 0.09, 0.1)),
        "case": pbr(stage, "/World/Looks/CaseShell", color=(0.04, 0.04, 0.045), roughness=0.45),
        "alu": pbr(stage, "/World/Looks/Alu", color=(0.72, 0.73, 0.75), roughness=0.3, metallic=1.0),
        "chrome": pbr(stage, "/World/Looks/Chrome", color=(0.8, 0.8, 0.82), roughness=0.18, metallic=1.0),
        "cable": pbr(stage, "/World/Looks/Cable", color=(0.02, 0.02, 0.02), roughness=0.5),
        "red": pbr(stage, "/World/Looks/WorklightRed", color=(0.12, 0.12, 0.13), roughness=0.4),
        "lens": pbr(stage, "/World/Looks/WorklightLens", color=(0.85, 0.85, 0.8), roughness=0.15),
    }
    floor = boxmesh(stage, "/World/Holo/floor", (0, 0.4, -0.01), (9.0, 8.4, 0.02),
                    uv_per_m=float(os.environ.get("HOLO_FLOOR_UV", "1.6")))  # 0.31 m tiles
    bind(floor.GetPrim(), M["grid"])
    for name, c, size in (("back", (0, 4.6, 1.6), (9.0, 0.02, 3.2)), ("left", (-4.5, 0.4, 1.6), (0.02, 8.4, 3.2)),
                          ("right", (4.5, 0.4, 1.6), (0.02, 8.4, 3.2))):
        bind(boxmesh(stage, f"/World/Holo/wall_{name}", c, size, uv_per_m=1.6).GetPrim(), M["wall"])
    if os.environ.get("HOLO_DARK"):  # dark set: close the open side, otherwise the dome shows as a grey band
        bind(boxmesh(stage, "/World/Holo/wall_front", (0, -3.8, 1.6), (9.0, 0.02, 3.2), uv_per_m=1.6).GetPrim(), M["wall"])
    # the centre circle is a ring of grey gaffer tape on the floor (spike mark, 1 mm, render-only)
    bind(tube(stage, "/World/Holo/centre_ring", 0.625, 0.675, 0.0, 0.001, seg=160, center=(1.0, -0.1, 0)).GetPrim(),
         M["spike"])
    for r in Y.RAMPS:
        mat = {"ramp_north": "plywood", "ramp_west": "carpet", "ramp_south": "metal"}[r[0]]
        for name, c, hs, q, role in Y.ramp_boxes(r):
            oriented_box(stage, K, f"/World/Holo/{name}", c, hs, q, 2.0, M[mat])
    for i, (x, y) in enumerate(Y.CONES):  # cone colliders -> stacks of gaffer tape
        for j in range(3):
            z0 = 0.047 * j
            bind(tube(stage, f"/World/Holo/tape{i}_{j}", 0.026, 0.05, z0, z0 + 0.045, seg=48, center=(x, y, 0)).GetPrim(),
                 M["tape_silver" if j == 1 else "tape"])
            bind(tube(stage, f"/World/Holo/tape{i}_{j}_core", 0.022, 0.026, z0, z0 + 0.045, seg=48,
                      center=(x, y, 0)).GetPrim(), M["tape_core"])
    for i, (x, y, yaw) in enumerate(Y.PLANTERS):  # planter colliders -> sandbags of the same size
        sx, sy, sz = Y.PLANTER_SIZE
        q = (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))
        oriented_box(stage, K, f"/World/Holo/sandbag{i}", (x, y, sz / 2), (sx / 2, sy / 2, sz / 2), q, 3.0, M["canvas"])
    # team-style props outside the walking area (render-only)
    def box(name, c, hs, yaw, mat, uv=2.0):
        q = (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))
        oriented_box(stage, K, f"/World/Props/{name}", c, hs, q, uv, M[mat])
    ab = (0.254, 0.1525, 0.1015)
    box("apple0", (-1.55, 3.05, ab[2]), ab, 0.12, "plywood")
    box("apple1", (-1.5, 3.07, 3 * ab[2] + 0.001), ab, 0.05, "plywood")
    box("apple2", (-0.95, 3.15, ab[0]), (ab[2], ab[1], ab[0]), -0.2, "plywood")
    box("case", (1.35, 3.2, 0.3), (0.4, 0.3, 0.3), -0.06, "case")
    for ix, dx in enumerate((-0.4, 0.4)):
        for iy, dy in enumerate((-0.3, 0.3)):
            c = (1.35 + dx * math.cos(-0.06) - dy * math.sin(-0.06), 3.2 + dx * math.sin(-0.06) + dy * math.cos(-0.06), 0.3)
            box(f"case_edge{ix}{iy}", c, (0.018, 0.018, 0.305), -0.06, "alu", 1.0)
    box("case_lid", (1.35, 3.2, 0.605), (0.405, 0.305, 0.006), -0.06, "alu", 1.0)
    box("worklight", (1.25, 3.18, 0.74), (0.11, 0.06, 0.13), 0.25, "red", 1.0)
    box("worklight_lens", (1.28, 3.115, 0.74), (0.085, 0.004, 0.1), 0.25, "lens", 1.0)
    bind(tube(stage, "/World/Props/xlr_coil", 0.09, 0.125, 0.0, 0.03, seg=64, center=(0.55, 3.15, 0)).GetPrim(), M["cable"])
    cx, cy = 3.05, 2.55
    for j, a in enumerate((0.3, 2.4, 4.5)):
        e = np.array([cx + 0.42 * math.cos(a), cy + 0.42 * math.sin(a), 0.02])
        m = np.array([cx, cy, 0.45])
        cap = UsdGeom.Capsule.Define(stage, f"/World/Props/cstand_leg{j}")
        cap.CreateRadiusAttr(0.011)
        cap.CreateHeightAttr(float(np.linalg.norm(m - e)))
        cap.CreateAxisAttr("Z")
        K.aim(cap.GetPrim(), (e + m) / 2, m + (m - e) * 10)
        bind(cap.GetPrim(), M["chrome"])
    for j, (a, b) in enumerate((((cx, cy, 0.3), (cx, cy, 1.8)), ((cx, cy, 1.55), (cx - 0.9, cy - 0.35, 1.7)))):
        a, b = np.array(a), np.array(b)
        cap = UsdGeom.Capsule.Define(stage, f"/World/Props/cstand_pole{j}")
        cap.CreateRadiusAttr(0.014 if j == 0 else 0.01)
        cap.CreateHeightAttr(float(np.linalg.norm(b - a)))
        cap.CreateAxisAttr("Z")
        K.aim(cap.GetPrim(), (a + b) / 2, b + (b - a) * 10)
        bind(cap.GetPrim(), M["chrome"])
    box("cstand_bag", (cx + 0.42 * math.cos(0.3) * 0.7, cy + 0.42 * math.sin(0.3) * 0.7, 0.06), (0.15, 0.12, 0.06), 0.3,
        "canvas", 3.0)
    # light: cool dome (team), soft key from the side at about 46 deg, front fill (about 3:1), soft rim from behind
    import os

    def rgb(env, default):
        v = os.environ.get(env)
        return tuple(float(x) for x in v.split(",")) if v else default
    dome = UsdLux.DomeLight.Define(stage, "/World/Lights/Dome")
    dome.CreateColorAttr(Gf.Vec3f(*rgb("HOLO_DOME_COLOR", (0.86, 0.91, 1.0))))
    dome.CreateIntensityAttr(float(dome_intensity))
    for name, pos, tgt, (w, h), inten, col in (
            ("Key", key_from, (0.6, 0.3, 0.0), (2.4, 1.6), key_intensity, rgb("HOLO_KEY_COLOR", (1.0, 0.98, 0.95))),
            ("Fill", (0.8, -3.6, 1.1), (0.6, 0.0, 0.15), (2.6, 1.2), fill_intensity, rgb("HOLO_FILL_COLOR", (0.95, 0.97, 1.0))),
            ("Rim", (0.9, 3.9, 1.7), (0.6, 0.6, 0.1), (2.0, 0.8), rim_intensity, rgb("HOLO_RIM_COLOR", (1.0, 1.0, 1.0)))):
        L = UsdLux.RectLight.Define(stage, f"/World/Lights/{name}")
        L.CreateWidthAttr(float(w))
        L.CreateHeightAttr(float(h))
        L.CreateIntensityAttr(float(inten))
        L.CreateColorAttr(Gf.Vec3f(*col))
        K.aim(L.GetPrim(), pos, tgt)
