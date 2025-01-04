# %%
from typing import Union
import cadquery as cq
import jax.numpy as jnp
from nebula.evaluators.bspline import BsplineSurface, normalize_knot_vector
from nebula.utils.occ_utils import OCCUtils
from OCP.TopoDS import TopoDS_Face


class OCPAdapter:

    @staticmethod
    def convert_face(face: Union[cq.Face, TopoDS_Face]):
        nurb_face = OCCUtils.convert_to_nurb_face(face.wrapped if isinstance(face, cq.Face) else face)
        geom_bspline_surface = OCCUtils.get_bspline_surface(nurb_face)

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



