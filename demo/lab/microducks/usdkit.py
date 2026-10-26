"""Small USD authoring helpers shared by the controller and the stage: meshes we cannot get from UsdGeom primitives
(rounded-rect prisms, textured quads and disks) and UsdPreviewSurface materials.

pxr is imported inside the functions so the module can be imported before Kit starts.
"""
from __future__ import annotations

import math
from pathlib import Path


# --------------------------------------------------------------------------------------------------------------- outlines
def rounded_rect(x0: float, x1: float, y0: float, y1: float, r: float, seg: int = 8) -> list[tuple[float, float]]:
    """CCW outline of a rounded rectangle."""
    r = min(r, (x1 - x0) / 2, (y1 - y0) / 2)
    pts: list[tuple[float, float]] = []
    corners = [(x1 - r, y0 + r, -90.0), (x1 - r, y1 - r, 0.0), (x0 + r, y1 - r, 90.0), (x0 + r, y0 + r, 180.0)]
    for cx, cy, a0 in corners:
        for i in range(seg + 1):
            a = math.radians(a0 + 90.0 * i / seg)
            pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return pts


def circle(r: float, seg: int = 48) -> list[tuple[float, float]]:
    """CCW outline of a circle."""
    return [(r * math.cos(2 * math.pi * i / seg), r * math.sin(2 * math.pi * i / seg)) for i in range(seg)]


def fluted(r: float, teeth: int = 20, depth: float = 0.12, per_tooth: int = 4) -> list[tuple[float, float]]:
    """CCW outline of a knob skirt with the usual moulded ribs, so a gripper has something to bite on."""
    pts, n = [], teeth * per_tooth
    for i in range(n):
        a = 2 * math.pi * i / n
        rr = r * (1.0 - depth * (0.5 - 0.5 * math.cos(teeth * a)))
        pts.append((rr * math.cos(a), rr * math.sin(a)))
    return pts


# --------------------------------------------------------------------------------------------------------------- meshes
def prism(stage, path: str, outline, z0: float, z1: float, cap_top: bool = True, cap_bottom: bool = True):
    """Extrude a closed 2-D outline between z0 and z1 (no UVs: for painted-colour parts)."""
    from pxr import Gf, UsdGeom, Vt

    n = len(outline)
    pts = [Gf.Vec3f(float(x), float(y), float(z0)) for x, y in outline]
    pts += [Gf.Vec3f(float(x), float(y), float(z1)) for x, y in outline]
    counts, idx = [], []
    for i in range(n):
        j = (i + 1) % n
        counts.append(4)
        idx += [i, j, j + n, i + n]
    if cap_top:
        counts.append(n)
        idx += [n + i for i in range(n)]
    if cap_bottom:
        counts.append(n)
        idx += [n - 1 - i for i in range(n)]
    m = UsdGeom.Mesh.Define(stage, path)
    m.CreatePointsAttr(Vt.Vec3fArray(pts))
    m.CreateFaceVertexCountsAttr(Vt.IntArray(counts))
    m.CreateFaceVertexIndicesAttr(Vt.IntArray(idx))
    m.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    m.CreateExtentAttr(UsdGeom.Mesh.ComputeExtentFromPlugins(m, 0.0) or [])
    return m


def quad(stage, path: str, x0: float, x1: float, y0: float, y1: float, z: float, uv=(0.0, 1.0, 0.0, 1.0)):
    """Single upward-facing textured quad. uv = (u0, u1, v0, v1) at (x0, x1, y0, y1)."""
    from pxr import Gf, Sdf, UsdGeom, Vt

    u0, u1, v0, v1 = uv
    m = UsdGeom.Mesh.Define(stage, path)
    m.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(x0, y0, z), Gf.Vec3f(x1, y0, z), Gf.Vec3f(x1, y1, z), Gf.Vec3f(x0, y1, z)]))
    m.CreateFaceVertexCountsAttr(Vt.IntArray([4]))
    m.CreateFaceVertexIndicesAttr(Vt.IntArray([0, 1, 2, 3]))
    m.CreateNormalsAttr(Vt.Vec3fArray([Gf.Vec3f(0, 0, 1)] * 4))
    m.SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)
    pv = UsdGeom.PrimvarsAPI(m).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying)
    pv.Set(Vt.Vec2fArray([Gf.Vec2f(u0, v0), Gf.Vec2f(u1, v0), Gf.Vec2f(u1, v1), Gf.Vec2f(u0, v1)]))
    m.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    return m


def disk(stage, path: str, r: float, z: float, seg: int = 96, uv_side: float | None = None):
    """Upward-facing disk; UVs map a square texture of side `uv_side` centred on the disk."""
    from pxr import Gf, Sdf, UsdGeom, Vt

    side = uv_side or (2 * r)
    pts = [Gf.Vec3f(0.0, 0.0, z)]
    uvs = [Gf.Vec2f(0.5, 0.5)]
    for i in range(seg):
        a = 2 * math.pi * i / seg
        x, y = r * math.cos(a), r * math.sin(a)
        pts.append(Gf.Vec3f(x, y, z))
        uvs.append(Gf.Vec2f(0.5 + x / side, 0.5 + y / side))
    counts, idx, fuv = [], [], []
    for i in range(seg):
        a, b = 1 + i, 1 + (i + 1) % seg
        counts.append(3)
        idx += [0, a, b]
        fuv += [uvs[0], uvs[a], uvs[b]]
    m = UsdGeom.Mesh.Define(stage, path)
    m.CreatePointsAttr(Vt.Vec3fArray(pts))
    m.CreateFaceVertexCountsAttr(Vt.IntArray(counts))
    m.CreateFaceVertexIndicesAttr(Vt.IntArray(idx))
    m.CreateNormalsAttr(Vt.Vec3fArray([Gf.Vec3f(0, 0, 1)] * len(idx)))
    m.SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)
    pv = UsdGeom.PrimvarsAPI(m).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying)
    pv.Set(Vt.Vec2fArray(fuv))
    m.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    return m


def tube(stage, path: str, r_in: float, r_out: float, z0: float, z1: float, seg: int = 96, center=(0.0, 0.0, 0.0),
         span=(0.0, 2 * math.pi)):
    """Z-axis annulus, optionally only part of a turn. A solid cylinder would cap over whatever sits inside it."""
    from pxr import Gf, UsdGeom, Vt

    a0, a1 = float(span[0]), float(span[1])
    closed = abs((a1 - a0) - 2 * math.pi) < 1e-9
    n = seg if closed else seg + 1                     # an open arc needs both end rings
    pts, counts, idx = [], [], []
    for r in (r_out, r_in):
        for z in (z0, z1):
            for i in range(n):
                a = a0 + (a1 - a0) * i / seg
                pts.append(Gf.Vec3f(float(r * math.cos(a)), float(r * math.sin(a)), float(z)))
    OB, OT, IB, IT = 0, n, 2 * n, 3 * n                # outer/inner, bottom/top rings
    for s0, s1, flip in ((OB, OT, False), (IB, IT, True), (OT, IT, False), (OB, IB, True)):
        for i in range(seg):
            j = (i + 1) % n
            q = [s0 + i, s0 + j, s1 + j, s1 + i]
            counts.append(4)
            idx += q[::-1] if flip else q
    if not closed:
        for i, flip in ((0, True), (seg, False)):      # flat faces where the arc is cut
            q = [OB + i, OT + i, IT + i, IB + i]
            counts.append(4)
            idx += q[::-1] if flip else q
    m = UsdGeom.Mesh.Define(stage, path)
    m.CreatePointsAttr(Vt.Vec3fArray(pts))
    m.CreateFaceVertexCountsAttr(Vt.IntArray(counts))
    m.CreateFaceVertexIndicesAttr(Vt.IntArray(idx))
    m.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    if any(center):
        UsdGeom.Xformable(m).AddTranslateOp().Set(Gf.Vec3d(*map(float, center)))
    return m


def cone(stage, path: str, r0: float, r1: float, length: float, seg: int = 28):
    """Open-ended cone along +Z from radius r0 at z=0 to r1 at z=length. Used for light beams, so no caps."""
    from pxr import Gf, UsdGeom, Vt

    pts, counts, idx = [], [], []
    for r, z in ((r0, 0.0), (r1, float(length))):
        for i in range(seg):
            a = 2 * math.pi * i / seg
            pts.append(Gf.Vec3f(float(r * math.cos(a)), float(r * math.sin(a)), z))
    for i in range(seg):
        j = (i + 1) % seg
        counts.append(4)
        idx += [i, j, seg + j, seg + i]
    m = UsdGeom.Mesh.Define(stage, path)
    m.CreatePointsAttr(Vt.Vec3fArray(pts))
    m.CreateFaceVertexCountsAttr(Vt.IntArray(counts))
    m.CreateFaceVertexIndicesAttr(Vt.IntArray(idx))
    m.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    m.CreateDoubleSidedAttr(True)
    return m


def place(prim, pos=(0.0, 0.0, 0.0), rot_xyz_deg=(0.0, 0.0, 0.0), scale=None):
    """Replace a prim's transform with translate, rotate XYZ, then optional scale.

    This clears whatever ops were there, so do not call it on a `box`: pass `rot_xyz_deg` to `box` instead, or the
    size it encodes as a scale op disappears. Meant for the meshes, whose shape lives in their points.
    """
    from pxr import Gf, UsdGeom

    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(*map(float, pos)))
    if any(rot_xyz_deg):
        xf.AddRotateXYZOp().Set(Gf.Vec3f(*map(float, rot_xyz_deg)))
    if scale is not None:
        xf.AddScaleOp().Set(Gf.Vec3f(*map(float, scale)))
    return prim


def look_at_quat_wxyz(eye, target, up=(0.0, 0.0, 1.0)) -> tuple:
    """Orientation whose local -Z points from `eye` at `target`.

    -Z is the axis UsdGeom.Camera looks down and the one UsdLux's disk and rect lights emit along, so the same
    quaternion aims a camera and a lamp.
    """
    import numpy as np

    eye, target, up = np.asarray(eye, float), np.asarray(target, float), np.asarray(up, float)
    f = target - eye
    f /= np.linalg.norm(f)
    if abs(float(np.dot(f, up))) > 0.999:
        up = np.array([0.0, 1.0, 0.0])
    r = np.cross(f, up)
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    m = np.stack([r, u, -f], axis=1)
    w = math.sqrt(max(1e-9, 1.0 + m[0, 0] + m[1, 1] + m[2, 2])) / 2
    return (w, (m[2, 1] - m[1, 2]) / (4 * w), (m[0, 2] - m[2, 0]) / (4 * w), (m[1, 0] - m[0, 1]) / (4 * w))


def aim(prim, pos, target, up=(0.0, 0.0, 1.0)):
    """Place `prim` at `pos` with its local -Z pointing at `target`. Same caveat as `place`: it clears the ops.

    Double precision throughout, because clearing the op order leaves the old attributes on the prim and re-adding one
    at a different precision than whoever created it is an error, not a conversion.
    """
    from pxr import Gf, UsdGeom

    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*map(float, pos)))
    w, x, y, z = look_at_quat_wxyz(pos, target, up)
    xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(w, x, y, z))
    return prim


def boxmesh(stage, path: str, center, size, uv_per_m: float = 1.0, color=None, rot_xyz_deg=None):
    """A box as an explicit mesh with UVs, so it can take a tiling texture.

    `box` is a `UsdGeom.Cube`, which has no UVs worth relying on -- an implicit surface's parameterisation is the
    renderer's business, and Hydra does not offer one. That is fine for a flat colour and useless for anything with
    grain, which is most of a stage: the floor, the riser and the desk are the three largest things on camera and a
    constant roughness over that much area is exactly what reads as computer graphics.

    UVs are in metres times `uv_per_m` rather than 0..1 per face, so one texture holds one physical scale wherever it
    is applied: the same scuffed-ply map on a 3.4 m riser and on a 0.4 m box gives grain the same size on both, which
    is what stops a small prop from looking like a photograph of a large one. Needs a material with `wrap="repeat"`.

    Built at the origin and placed with a transform, same as `box`, rather than with its vertices written out at
    `center`. The difference only shows up once something is rotated -- vertices in world coordinates turn about the
    world origin, which sends a speaker cabinet tilted 7 degrees off across the room instead of nodding in place.
    """
    from pxr import Gf, Sdf, UsdGeom, Vt

    hx, hy, hz = (float(v) / 2 for v in size)
    # Per face: the four corners, and which two world axes become u and v. Spans are the physical size of the face
    # along those axes, so a tall thin face gets a tall thin piece of the texture rather than a stretched square.
    faces = [
        ((-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1), (hx, hy)),      # bottom
        ((-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1), (hx, hy)),          # top
        ((-1, -1, -1), (1, -1, -1), (1, -1, 1), (-1, -1, 1), (hx, hz)),      # -y
        ((-1, 1, -1), (1, 1, -1), (1, 1, 1), (-1, 1, 1), (hx, hz)),          # +y
        ((-1, -1, -1), (-1, 1, -1), (-1, 1, 1), (-1, -1, 1), (hy, hz)),      # -x
        ((1, -1, -1), (1, 1, -1), (1, 1, 1), (1, -1, 1), (hy, hz)),          # +x
    ]
    pts, counts, idx, uvs = [], [], [], []
    for *corners, (su, sv) in faces:
        counts.append(4)
        for k, (sx, sy, sz) in enumerate(corners):
            pts.append(Gf.Vec3f(sx * hx, sy * hy, sz * hz))
            idx.append(len(pts) - 1)
            u, v = ((0, 0), (1, 0), (1, 1), (0, 1))[k]
            uvs.append(Gf.Vec2f(float(u * 2 * su * uv_per_m), float(v * 2 * sv * uv_per_m)))
    m = UsdGeom.Mesh.Define(stage, path)
    m.CreatePointsAttr(Vt.Vec3fArray(pts))
    m.CreateFaceVertexCountsAttr(Vt.IntArray(counts))
    m.CreateFaceVertexIndicesAttr(Vt.IntArray(idx))
    pv = UsdGeom.PrimvarsAPI(m).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying)
    pv.Set(Vt.Vec2fArray(uvs))
    m.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    place(m.GetPrim(), pos=center, rot_xyz_deg=rot_xyz_deg or (0.0, 0.0, 0.0))
    if color is not None:
        m.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    return m


def box(stage, path: str, center, size, color=None, rot_xyz_deg=None):
    from pxr import Gf, UsdGeom

    c = UsdGeom.Cube.Define(stage, path)
    c.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(c)
    xf.AddTranslateOp().Set(Gf.Vec3d(*map(float, center)))
    if rot_xyz_deg is not None and any(rot_xyz_deg):
        xf.AddRotateXYZOp().Set(Gf.Vec3f(*map(float, rot_xyz_deg)))
    xf.AddScaleOp().Set(Gf.Vec3f(*map(float, size)))
    if color is not None:
        c.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    return c


def cylinder(stage, path: str, r: float, h: float, center=(0, 0, 0), seg: int = 48, outline=None, color=None,
             rot_xyz_deg=None):
    """Z-axis cylinder as an explicit mesh.

    Not UsdGeom.Cylinder on purpose: Hydra tessellates implicit surfaces with ten radial slices, which on a 43 mm jog
    platter reads as a visible decagon.

    Pass `rot_xyz_deg` to lay one on its side, and note what it changes: the cylinder is then built about the origin
    and `center` becomes a transform, because rotating a mesh whose vertices are already out at `center` swings it
    about the world origin instead of its own axis. Without a rotation nothing moves and the vertices stay where they
    were, which is how every existing caller expects it to behave.
    """
    from pxr import Gf, UsdGeom

    cx, cy, cz = map(float, center)
    turn = rot_xyz_deg is not None and any(rot_xyz_deg)
    z = 0.0 if turn else cz
    m = prism(stage, path, outline if outline is not None else circle(r, seg), z - h / 2, z + h / 2)
    if turn:
        place(m.GetPrim(), pos=(cx, cy, cz), rot_xyz_deg=rot_xyz_deg)
    else:
        UsdGeom.Xformable(m).AddTranslateOp().Set(Gf.Vec3d(cx, cy, 0.0))
    if color is not None:
        m.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    return m


def frustum(stage, path: str, r0: float, r1: float, z0: float, z1: float, seg: int = 96, center=(0, 0, 0),
            cap_bottom: bool = True, cap_top: bool = True, color=None):
    """Z-axis truncated cone, radius r0 at z0 tapering to r1 at z1.

    A real jog platter is not a cylinder: the top face is a few millimetres narrower than the base, which is what
    stops a thumb sliding off the edge. Convex, so one of these also serves as its own convexHull collider.
    """
    from pxr import Gf, UsdGeom, Vt

    cx, cy, cz = map(float, center)
    pts, counts, idx = [], [], []
    for r, z in ((r0, z0), (r1, z1)):
        for i in range(seg):
            a = 2 * math.pi * i / seg
            pts.append(Gf.Vec3f(float(cx + r * math.cos(a)), float(cy + r * math.sin(a)), float(cz + z)))
    for i in range(seg):
        j = (i + 1) % seg
        counts.append(4)
        idx += [i, j, seg + j, seg + i]
    if cap_bottom:
        pts.append(Gf.Vec3f(cx, cy, float(cz + z0)))
        c = len(pts) - 1
        for i in range(seg):
            counts.append(3)
            idx += [c, (i + 1) % seg, i]
    if cap_top:
        pts.append(Gf.Vec3f(cx, cy, float(cz + z1)))
        c = len(pts) - 1
        for i in range(seg):
            counts.append(3)
            idx += [c, seg + i, seg + (i + 1) % seg]
    m = UsdGeom.Mesh.Define(stage, path)
    m.CreatePointsAttr(Vt.Vec3fArray(pts))
    m.CreateFaceVertexCountsAttr(Vt.IntArray(counts))
    m.CreateFaceVertexIndicesAttr(Vt.IntArray(idx))
    m.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    if color is not None:
        m.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    return m


# --------------------------------------------------------------------------------------------------------------- materials
def surface(stage, path: str, color=(0.05, 0.05, 0.05), roughness: float = 0.5, metallic: float = 0.0,
            albedo_tex: str | Path | None = None, rough_tex: str | Path | None = None, emissive=None,
            opacity: float = 1.0, ior: float = 1.5, wrap: str = "clamp", tint=None,
            emissive_tex: str | Path | None = None, emissive_gain=None):
    """UsdPreviewSurface with optional albedo/roughness textures (st primvar reader).

    `wrap` is "clamp" for a texture laid out once over a part, which is how the controller's artwork works, and
    "repeat" for a tiling map on UVs measured in metres, which is how the set's surfaces work. `tint` multiplies an
    albedo texture, so one grey scuff map can serve a black floor and a grey riser.
    """
    from pxr import Gf, Sdf, UsdShade

    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, f"{path}/Surface")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(float(metallic))
    sh.CreateInput("useSpecularWorkflow", Sdf.ValueTypeNames.Int).Set(0)
    sh.CreateInput("ior", Sdf.ValueTypeNames.Float).Set(float(ior))
    if opacity < 1.0:
        sh.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(float(opacity))
        # Without a zero threshold the translator treats the surface as a cutout and it comes out fully opaque.
        sh.CreateInput("opacityThreshold", Sdf.ValueTypeNames.Float).Set(0.0)
    if emissive is not None:
        sh.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*emissive))
    reader = None
    if albedo_tex or rough_tex or emissive_tex:
        reader = UsdShade.Shader.Define(stage, f"{path}/stReader")
        reader.CreateIdAttr("UsdPrimvarReader_float2")
        reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    if albedo_tex:
        t = UsdShade.Shader.Define(stage, f"{path}/AlbedoTex")
        t.CreateIdAttr("UsdUVTexture")
        t.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(str(albedo_tex))
        t.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
        t.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set(wrap)
        t.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set(wrap)
        if tint is not None:
            t.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(*tint, 1.0))
        t.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
        t.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(t.ConnectableAPI(), "rgb")
    else:
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    if rough_tex:
        t = UsdShade.Shader.Define(stage, f"{path}/RoughTex")
        t.CreateIdAttr("UsdUVTexture")
        t.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(str(rough_tex))
        t.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("raw")
        t.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set(wrap)
        t.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set(wrap)
        t.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
        t.CreateOutput("r", Sdf.ValueTypeNames.Float)
        sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).ConnectToSource(t.ConnectableAPI(), "r")
    else:
        sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(roughness))
    if emissive_tex:
        # A sign is a picture that emits, not a picture that is lit, so the same image drives both diffuse and
        # emissive. `emissive_gain` scales it above 1 where the thing has to read as a light source rather than as
        # a bright surface, which an LED wall in a dark room does.
        t = UsdShade.Shader.Define(stage, f"{path}/EmitTex")
        t.CreateIdAttr("UsdUVTexture")
        t.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(str(emissive_tex))
        t.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
        t.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set(wrap)
        t.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set(wrap)
        if emissive_gain is not None:
            t.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(*emissive_gain, 1.0))
        t.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
        t.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        sh.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(t.ConnectableAPI(), "rgb")
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return mat


def bind(prim, mat) -> None:
    from pxr import UsdShade

    UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat)


# --------------------------------------------------------------------------------------------------------------- physics
def physics_material(stage, path: str, static: float = 1.1, dynamic: float = 0.95, restitution: float = 0.02):
    """A rubbing surface. Without one a collider gets the engine's default, and here that default has no friction at
    all, so a gripper can pinch a knob and twist 40 degrees without the knob turning a thousandth of a radian."""
    from pxr import UsdPhysics, UsdShade

    mat = UsdShade.Material.Define(stage, path)
    api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    api.CreateStaticFrictionAttr().Set(float(static))
    api.CreateDynamicFrictionAttr().Set(float(dynamic))
    api.CreateRestitutionAttr().Set(float(restitution))
    return mat


def rub(prim, mat) -> None:
    """Bind a physics material to a collider. The purpose is what keeps it from also being read as a surface shader."""
    from pxr import UsdShade

    UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat, UsdShade.Tokens.weakerThanDescendants, "physics")


def rigid_body(stage, path: str, mass: float, inertia, com=(0.0, 0.0, 0.0)):
    from pxr import Gf, UsdGeom, UsdPhysics

    x = UsdGeom.Xform.Define(stage, path)
    UsdPhysics.RigidBodyAPI.Apply(x.GetPrim())
    m = UsdPhysics.MassAPI.Apply(x.GetPrim())
    m.CreateMassAttr(float(mass))
    m.CreateDiagonalInertiaAttr(Gf.Vec3f(*map(float, inertia)))
    m.CreateCenterOfMassAttr(Gf.Vec3f(*map(float, com)))
    return x


def box_inertia(mass: float, sx: float, sy: float, sz: float) -> tuple[float, float, float]:
    k = mass / 12.0
    return (k * (sy * sy + sz * sz), k * (sx * sx + sz * sz), k * (sx * sx + sy * sy))


def cyl_inertia(mass: float, r: float, h: float) -> tuple[float, float, float]:
    ix = mass * (3 * r * r + h * h) / 12.0
    return (ix, ix, mass * r * r / 2.0)


def collider(prim, approximation: str | None = None) -> None:
    from pxr import UsdPhysics

    UsdPhysics.CollisionAPI.Apply(prim)
    if approximation:
        mc = UsdPhysics.MeshCollisionAPI.Apply(prim)
        mc.CreateApproximationAttr(approximation)


def revolute(stage, path: str, body0: str, body1: str, axis: str, pos0, pos1, limits=None, drive=None):
    """Revolute joint; limits in degrees (None = free spin). drive = (stiffness, damping, target) in SI/rad."""
    from pxr import Gf, Sdf, UsdPhysics

    j = UsdPhysics.RevoluteJoint.Define(stage, path)
    j.CreateBody0Rel().SetTargets([Sdf.Path(body0)])
    j.CreateBody1Rel().SetTargets([Sdf.Path(body1)])
    j.CreateAxisAttr(axis)
    j.CreateLocalPos0Attr(Gf.Vec3f(*map(float, pos0)))
    j.CreateLocalPos1Attr(Gf.Vec3f(*map(float, pos1)))
    q = Gf.Quatf(1.0, 0.0, 0.0, 0.0)
    j.CreateLocalRot0Attr(q)
    j.CreateLocalRot1Attr(q)
    if limits is not None:
        j.CreateLowerLimitAttr(float(limits[0]))
        j.CreateUpperLimitAttr(float(limits[1]))
    if drive is not None:
        kp, kd, target = drive
        d = UsdPhysics.DriveAPI.Apply(j.GetPrim(), "angular")
        d.CreateTypeAttr("force")
        d.CreateStiffnessAttr(float(kp) * math.pi / 180.0)
        d.CreateDampingAttr(float(kd) * math.pi / 180.0)
        d.CreateTargetPositionAttr(float(math.degrees(target)))
    return j


def prismatic(stage, path: str, body0: str, body1: str, axis: str, pos0, pos1, limits, drive=None):
    """Prismatic joint; limits in metres along `axis`. drive = (stiffness N/m, damping N s/m, target m)."""
    from pxr import Gf, Sdf, UsdPhysics

    j = UsdPhysics.PrismaticJoint.Define(stage, path)
    j.CreateBody0Rel().SetTargets([Sdf.Path(body0)])
    j.CreateBody1Rel().SetTargets([Sdf.Path(body1)])
    j.CreateAxisAttr(axis)
    j.CreateLocalPos0Attr(Gf.Vec3f(*map(float, pos0)))
    j.CreateLocalPos1Attr(Gf.Vec3f(*map(float, pos1)))
    q = Gf.Quatf(1.0, 0.0, 0.0, 0.0)
    j.CreateLocalRot0Attr(q)
    j.CreateLocalRot1Attr(q)
    j.CreateLowerLimitAttr(float(limits[0]))
    j.CreateUpperLimitAttr(float(limits[1]))
    if drive is not None:
        kp, kd, target = drive
        d = UsdPhysics.DriveAPI.Apply(j.GetPrim(), "linear")
        d.CreateTypeAttr("force")
        d.CreateStiffnessAttr(float(kp))
        d.CreateDampingAttr(float(kd))
        d.CreateTargetPositionAttr(float(target))
    return j


def filter_pairs(prim, others: list[str]) -> None:
    from pxr import Sdf, UsdPhysics

    api = UsdPhysics.FilteredPairsAPI.Apply(prim)
    rel = api.CreateFilteredPairsRel()
    rel.SetTargets([Sdf.Path(p) for p in others])
