from dataclasses import dataclass, field
from typing import Optional, Sequence, Union
import jax
import numpy as np
from scipy.spatial import Delaunay
import jax.numpy as jnp
from nebula.evaluators.bspline import BSplineEvaluator, BsplineSurface, get_sampling
from tqdm import tqdm

@dataclass
class Mesh:
    vertices: jnp.ndarray
    simplices: jnp.ndarray

    @staticmethod
    def empty():
        return Mesh(
            vertices=jnp.empty((0, 3)),
            simplices=jnp.empty((0, 3), dtype=jnp.int32),
        )

    @staticmethod
    def combine(meshes: list["Mesh"]):
        vertices = jnp.empty((0, 3))
        simplices = jnp.empty((0, 3), dtype=jnp.int32)
        for mesh in meshes:
            simplices = jnp.concatenate([simplices, mesh.simplices + len(vertices)])
            vertices = jnp.concatenate([vertices, mesh.vertices])
        return Mesh(vertices, simplices)


class Tesselator:

    @staticmethod
    def get_tri_normal(tri_vertices: Union[jnp.ndarray, np.ndarray]) -> jnp.ndarray:
        # Calculate two vectors along the edges of the triangle
        normal = jnp.cross(
            tri_vertices[1] - tri_vertices[0], tri_vertices[2] - tri_vertices[0]
        )
        unit_normal = normal / jnp.linalg.norm(normal, axis=0, keepdims=True)
        return unit_normal

    @staticmethod
    @jax.jit
    def get_bspline_tri_index(i: jnp.ndarray, j: jnp.ndarray, num_v: jnp.ndarray):
        index = jnp.array(
            [
                j + (i * num_v),
                j + ((i + 1) * num_v),
                j + 1 + ((i + 1) * num_v),
                j + 1 + (i * num_v),
            ],
            dtype=jnp.int32,
        )
        # + curr_index

        return jax.vmap(
            lambda tri_index, i: jnp.array(
                # [tri_index[i + 1], tri_index[i], tri_index[0]],
                [tri_index[0], tri_index[i], tri_index[i + 1]],
                dtype=jnp.int32,
            ),
            in_axes=(None, 0),
        )(index, jnp.arange(1, 3))

    @staticmethod
    def get_bspline_mesh_simplices(num_u: int, num_v: int):
        i, j = jnp.arange(num_u - 1), jnp.arange(num_v - 1)
        simplicies = jax.vmap(
            jax.vmap(
                Tesselator.get_bspline_tri_index,
                in_axes=(None, 0, None),
            ),
            in_axes=(0, None, None),
        )(i, j, jnp.array(num_v)).reshape(-1, 3)

        return simplicies

    @staticmethod
    def tesselate_surface(
        surf: BsplineSurface,
        u: Optional[jax.Array] = None,
        v: Optional[jax.Array] = None,
    ):
        u = get_sampling(0.0, 1.0, 20) if u is None else u
        v = get_sampling(0.0, 1.0, 20) if v is None else v
        simplicies = Tesselator.get_bspline_mesh_simplices(len(u), len(v))
        vertices = BSplineEvaluator.eval_surface(surf, u, v).reshape(-1, 3)
        return Mesh(vertices, simplicies)

    @staticmethod
    def tesselate(surfs: list[BsplineSurface]):
        meshes: list[Mesh] = []
        for surf in tqdm(surfs):
            mesh = Tesselator.tesselate_surface(surf)
            meshes.append(mesh)
        return Mesh.combine(meshes)
