# %%
from typing import Union, cast
from OCP.TColgp import TColgp_HArray1OfPnt, TColgp_HArray2OfPnt
from OCP.TColStd import TColStd_HArray1OfReal, TColStd_HArray1OfInteger
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace
from OCP.Geom import Geom_BSplineSurface
from OCP.gp import gp_Pnt
from OCP.TopoDS import TopoDS
import cadquery as cq
from cadquery.cq import CQObject
import jax
from jupyter_cadquery import show
import jax.numpy as jnp
from nebula.evaluators.bspline import BsplineSurface, normalize_knot_vector
from OCP.BRep import BRep_Builder
from OCP.TopoDS import TopoDS_Shape
from jupyter_cadquery import show
from OCP.BRepBuilderAPI import BRepBuilderAPI_NurbsConvert
from OCP.BRep import BRep_Tool, BRep_Builder
from OCP.GeomConvert import GeomConvert



class OCPAdapter:
    @staticmethod
    def import_step(file_name: str):
        workplane = cq.importers.importStep(file_name)
        return cast(Union[cq.Compound, cq.Solid], workplane.val())

    @staticmethod
    def to_assembly(compound: Union[cq.Compound, cq.Solid]):
        assembly = cq.Assembly()
        faces = compound.Faces()
        for i, face in enumerate(faces):
            assembly.add(face, name=f"Face{i}")
        return assembly

    @staticmethod
    def to_bspline_surface(face: cq.Face):

        nurbs_converter = BRepBuilderAPI_NurbsConvert(face.wrapped)
        nurbs_converter.Perform(face.wrapped)
        nurbs_shape = TopoDS.Face_s(nurbs_converter.Shape())
        brep_surface = BRep_Tool.Surface_s(nurbs_shape)
        geom_bspline_surface = GeomConvert.SurfaceToBSplineSurface_s(brep_surface)

        geom_bspline_surface.SetUNotPeriodic()
        geom_bspline_surface.SetVNotPeriodic()

        poles = geom_bspline_surface.Poles()
        v_knots = geom_bspline_surface.VKnots()
        u_knots = geom_bspline_surface.UKnots()

        v_multiplicities = geom_bspline_surface.VMultiplicities()
        u_multiplicities = geom_bspline_surface.UMultiplicities()

        row_length = poles.NbRows()
        column_length = poles.NbColumns()

        poles_arr: list[list[float]] = []
        for i in range(row_length):
            poles_arr.append([])
            for j in range(column_length):
                value = poles.Value(i + 1, j + 1)
                poles_arr[-1].append([value.X(), value.Y(), value.Z()])

        v_knots_arr = []
        for i in range(v_knots.Length()):
            v_knots_arr += [v_knots.Value(i + 1)] * v_multiplicities.Value(i + 1)
        u_knots_arr = []
        for i in range(u_knots.Length()):
            u_knots_arr += [u_knots.Value(i + 1)] * u_multiplicities.Value(i + 1)

        return BsplineSurface(
            ctrl_pnts=jnp.array(poles_arr),
            u_knots=normalize_knot_vector(u_knots_arr),
            v_knots=normalize_knot_vector(v_knots_arr),
            u_degree=geom_bspline_surface.UDegree(),
            v_degree=geom_bspline_surface.VDegree(),
        )


face_compound = OCPAdapter.import_step(
    "/workspaces/nebulaNew/CNCRD.STEP"
)
# assembly = OCPAdapter.to_assembly(face_compound)
# show(assembly)
# %%
filtered_faces: list[cq.Face] = []
for i, face in enumerate(face_compound.Faces()):
    if i in [387, 411, 388, 355, 115, 95, 113, 114, 111, 112, 428, 427, 106, 120]:
        filtered_faces.append(face)
    # face.exportBrep(f"face_{i}.brep")
# for i in [388, 412, 389, 356]:
#     builder = BRep_Builder()
#     shape = TopoDS_Shape()
#     return_code = BRepTools.Read_s(shape, f"face_{i-1}.brep", builder)
#     if return_code is False:
#         raise ValueError("Import failed, check file name")
#     face = cq.Compound(shape).Faces()[0]
#     filtered_faces.append(face)
# show(cq.Workplane().newObject([filtered_faces[0]]))

surfs = [OCPAdapter.to_bspline_surface(face) for face in filtered_faces]

pnts = jnp.empty((0, 3))
for surf in surfs:
    pnts = jnp.concatenate([pnts, surf.ctrl_pnts.reshape(-1,3)])
    # plotly plot points

import plotly.graph_objects as go
fig = go.Figure(data=[go.Scatter3d(x=pnts[:, 0].flatten(), y=pnts[:, 1].flatten(), z=pnts[:, 2].flatten(), mode="markers")])
fig.update_layout(scene=dict(aspectmode='data'))
fig.show()


# %%
from geomdl import BSpline
from geomdl.visualization import VisPlotly

def display_bspline_surf(bspline_data: BsplineSurface):
    # Create a BSpline surface instance
    surf = BSpline.Surface()

    # Set degrees
    surf.degree_u = bspline_data.u_degree
    surf.degree_v = bspline_data.v_degree

    # Get the control points from the generated grid
    surf.ctrlpts2d = bspline_data.ctrl_pnts.tolist()

    # Set knot vectors
    surf.knotvector_u = bspline_data.u_knots.tolist()
    surf.knotvector_v = bspline_data.v_knots.tolist()


    # Set sample size
    surf.sample_size = 100


    # Set visualization component
    surf.evaluate
    surf.vis = VisPlotly.VisSurface(legend=False)

    # print(surf.sample_size_u)
    # print(surf.sample_size_v)
    # print(surf.evalpts)
    # print(len(surf.evalpts), len(surf.evalpts[0]))

    # Plot the surface
    surf.render()

display_bspline_surf(surfs[-2])

# %%
from nebula.evaluators.bspline import BSplineEvaluator
res = BSplineEvaluator.eval_surface(surfs[-2], jnp.linspace(0,1,100), jnp.linspace(0,1,100))

# %%
from nebula.render.tesselation import Tesselator
from nebula.render.visualization import show as nebShow

mesh = Tesselator.tesselate(surfs)
nebShow(mesh, type="plot")
# rebuild = ShapeBuild_ReShape


# new_surface = Geom_BSplineSurface(bspline_surface.Poles(), u_knots, v_knots, u_multiplicities, v_multiplicities, u_degree, v_degree, u_periodic, v_periodic)
# new_face = 	BRepBuilderAPI_MakeFace(new_surface, 1e-6).Face()

# rebuild.Replace(face.wrapped, theNewFace, Standard_False)
# shcmpnd = rebuild.Apply(cmpnd, TopAbs_SHAPE, 1)


#     @staticmethod
#     def to_topo_face(face: Face, tolerance: float = 1e-6):
#         geom_surface = OCPAdapter.to_geom_surface(face.geom)
#         return BRepBuilderAPI_MakeFace(geom_surface, tolerance)

#     @staticmethod
#     def to_geom_surface(geom: GeomSurface):
#         if isinstance(geom, BSplineSurface):
#             return OCPAdapter.to_geom_bspline_surface(geom)
#         else:
#             raise NotImplementedError

#     @staticmethod
#     def to_geom_bspline_surface(bspline_surface: BSplineSurface):
#         poles = OCPAdapter.to_pnt_array2(bspline_surface.ctrl_pnts)
#         u_knots = OCPAdapter.to_real_array(bspline_surface.u_knots)
#         v_knots = OCPAdapter.to_real_array(bspline_surface.v_knots)

#         u_mults = OCPAdapter.to_int_array(bspline_surface.u_mults)
#         v_mults = OCPAdapter.to_int_array(bspline_surface.v_mults)

#         u_degree = bspline_surface.u_degree
#         v_degree = bspline_surface.v_degree

#         return Geom_BSplineSurface(poles, u_knots, v_knots, u_mults, v_mults, u_degree, v_degree)

#     @staticmethod
#     def to_pnt_array2(vector_tensor: torch.Tensor):
#         arr = TColgp_HArray2OfPnt(1, len(vector_tensor), 1, len(vector_tensor[0]))
#         for i in range(len(vector_tensor)):
#             for j, vector in enumerate(vector_tensor[i]):
#                 pnt = gp_Pnt(float(vector[0]), float(vector[1]), float(vector[2]))
#                 arr.SetValue(i + 1, j + 1, pnt)
#         return arr

#     @staticmethod
#     def to_pnt_array(vector_tensor: torch.Tensor):
#         arr = TColgp_HArray1OfPnt(1, len(vector_tensor))
#         for i, vector in enumerate(vector_tensor):
#             pnt = gp_Pnt(float(vector[0]), float(vector[1]), float(vector[2]))
#             arr.SetValue(i + 1, pnt)
#         return arr

#     @staticmethod
#     def to_real_array(real_tensor: torch.Tensor):
#         arr = TColStd_HArray1OfReal(1, len(real_tensor))
#         for i, real in enumerate(real_tensor):
#             arr.SetValue(i + 1, float(real))
#         return arr

#     @staticmethod
#     def to_int_array(int_tensor: torch.Tensor):
#         arr = TColStd_HArray1OfInteger(1, len(int_tensor))
#         for i, real in enumerate(int_tensor):
#             arr.SetValue(i + 1, int(real))
#         return arr
# %%
