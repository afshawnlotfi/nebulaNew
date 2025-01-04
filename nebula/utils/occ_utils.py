from typing import Union, cast
import cadquery as cq
from OCP.TopoDS import TopoDS
from OCP.BRepBuilderAPI import BRepBuilderAPI_NurbsConvert
from OCP.GeomConvert import GeomConvert
from OCP.BRep import BRep_Tool
from OCP.gp import gp_Pnt
from OCP.GeomAPI import GeomAPI_ProjectPointOnSurf
from OCP.Geom import Geom_BSplineSurface
from OCP.Bnd import Bnd_Box
from OCP.BRepBndLib import BRepBndLib
from OCP.BRep import BRep_Tool
from OCP.TopoDS import TopoDS_Face

import jax.numpy as jnp


class OCCUtils:
    @staticmethod
    def convert_to_nurb_face(face: TopoDS_Face):
        nurbs_converter = BRepBuilderAPI_NurbsConvert(face)
        nurbs_converter.Perform(face)
        return TopoDS.Face_s(nurbs_converter.Shape())

    @staticmethod
    def get_bspline_surface(face: TopoDS_Face):
        brep_surface = BRep_Tool.Surface_s(face)
        geom_bspline_surface = GeomConvert.SurfaceToBSplineSurface_s(brep_surface)

        geom_bspline_surface.SetUNotPeriodic()
        geom_bspline_surface.SetVNotPeriodic()

        return geom_bspline_surface


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
    def project_points(points: jnp.ndarray, face: TopoDS_Face):
        # Create bounding box
        bbox = Bnd_Box()
        BRepBndLib.Add_s(face, bbox)
        xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
        surface = OCCUtils.get_bspline_surface(face)
        projected_knots = []
        for point in points:
            # Check if point is within bounding box
            x, y, z = float(point[0]), float(point[1]), float(point[2])
            # if (xmin <= x <= xmax and ymin <= y <= zmax and  zmin <= z <= zmax):
                # Convert point to OpenCASCADE point
            occ_point = gp_Pnt(x, y, z)
            
            # Create projector
            projector = GeomAPI_ProjectPointOnSurf(occ_point, surface)
            
            if projector.NbPoints() > 0:
                # Get UV parameters of projected point
                u, v = projector.Parameters(1)
                projected_knots.append([u, v])
            
        return jnp.array(projected_knots)
