
import jax
import jax.numpy as jnp
from typing import Callable, Optional, Sequence, Union
import jax_dataclasses as jdc
from jaxtyping import Array, Float


SpanFunction = Callable[[int, jax.Array, int, jax.Array], jax.Array]


def get_sampling(start: float, end: float, num_points: int, is_cosine_sampling: bool = False):
    if is_cosine_sampling:
        beta = jnp.linspace(0.0,jnp.pi, num_points, endpoint=True)
        return 0.5*(1.0-jnp.cos(beta))
    return jnp.linspace(start, end, num_points, endpoint=True)

def normalize_knot_vector(knot_vector: Union[jax.Array, Sequence[float]], decimals: int = 18):
    knot_vector = jnp.array(knot_vector)
    first_knot = knot_vector[0]
    last_knot = knot_vector[-1]
    denominator = last_knot - first_knot
    
    return jnp.round((knot_vector - first_knot) / denominator, decimals=decimals)

@jdc.pytree_dataclass
class BsplineSurface:
    ctrl_pnts: Float[Array, "pnts xyz"]
    u_knots: jax.Array
    v_knots: jax.Array
    u_degree: int
    v_degree: int


@jdc.pytree_dataclass
class BsplineCurve:
    ctrl_pnts: Float[Array, "pnts xyz"]
    degree: int
    knots: Optional[jax.Array] = None
    

# def find_span_linear(degree: int, knot_vector: jnp.ndarray, num_ctrlpts: int, knot: float):
#     """ Finds the span of a single knot over the knot vector using linear search.

#     Alternative implementation for the Algorithm A2.1 from The NURBS Book by Piegl & Tiller.

#     :param degree: degree, :math:`p`
#     :type degree: int
#     :param knot_vector: knot vector, :math:`U`
#     :type knot_vector: list, tuple
#     :param num_ctrlpts: number of control points, :math:`n + 1`
#     :type num_ctrlpts: int
#     :param knot: knot or parameter, :math:`u`
#     :type knot: float
#     :return: knot span
#     :rtype: int
#     """
#     span = degree + 1  # Knot span index starts from zero
#     while span < num_ctrlpts and knot_vector[span] <= knot:
#         span += 1

#     return span - 1


class BSplineEvaluator:
    @staticmethod
    def find_spans(
        degree: int,
        knot_vector: jax.Array,
        num_ctrlpts: int,
        knot_samples: jax.Array,
    ):
        """Finds the span of a single knot over the knot vector using linear search.

        Alternative implementation for the Algorithm A2.1 from The NURBS Book by Piegl & Tiller.

        :param degree: degree, :math:`p`
        :type degree: jax.Array, (1,)
        :param knot_vector: knot vector, :math:`U`
        :type knot_vector: torch.Tensor
        :param num_ctrlpts: number of control points, :math:`n + 1`
        :type num_ctrlpts: int
        :param knot: knot or parameter, :math:`u`
        :type knot: float
        :return: knot span
        :rtype: int
        """
        span_start = degree + 1
        span_offset = jnp.sum(
            jnp.expand_dims(knot_samples, axis=-1) > knot_vector[span_start:], axis=-1
        )
        span = jnp.clip(span_start + span_offset, a_max=num_ctrlpts)
        return span - 1
        # spans = jnp.zeros_like(knot_samples, dtype=jnp.int32)
        # for i, knot in enumerate(knot_samples):
        #     span = find_span_linear(degree, knot_vector, num_ctrlpts, knot)

        #     spans = spans.at[i].set(span)
        # return spans

    @staticmethod
    def basis_functions(
        degree: int,
        knot_vector: jax.Array,
        span: jax.Array,
        knot_samples: jax.Array,
    ):
        """Computes the non-vanishing basis functions for a single parameter.

        Implementation of Algorithm A2.2 pg 70 from The NURBS Book by Piegl & Tiller.
        Uses recurrence to compute the basis functions, also known as Cox - de
        Boor recursion formula.

        :param degree: degree, :math:`p`
        :type degree: jax.Array (1,)
        :param knot_vector: knot vector, :math:`U`
        :type knot_vector: list, tuple
        :param span: knot span, :math:`i`
        :type span: int
        :param knot: knot or parameter, :math:`u`
        :type knot: float
        :return: basis functions
        :rtype: list
        """
        N = jnp.ones((degree + 1, len(knot_samples)))
        left = jnp.expand_dims(knot_samples, axis=0) - knot_vector[span + 1 - jnp.arange(degree + 1)[:, None]]
        right = knot_vector[span + jnp.arange(degree + 1)[:, None]] - jnp.expand_dims(knot_samples, axis=0)

        def inner_body_fun(r, init_value):
            j, saved, new_N = init_value
            temp = new_N[r] / (right[r + 1] + left[j - r])
            next_N = new_N.at[r].set(saved + right[r + 1] * temp)
            saved = left[j - r] * temp
            return j, saved, next_N

        def outer_body_fun(j, N: jax.Array):
            saved = jnp.zeros(len(knot_samples))
            _, saved, N = jax.lax.fori_loop(0, j, inner_body_fun, (j, saved, N))
            return N.at[j].set(saved)

        return jax.lax.fori_loop(1, degree+1, outer_body_fun, N)

    @staticmethod
    def generate_line_knots():
        return BSplineEvaluator.generate_clamped_knots(1, 2)

    @staticmethod
    def generate_clamped_knots(degree: int, num_ctrlpts: int):
        """Generates a clamped knot vector.

        :param degree: non-zero degree of the curve
        :type degree: int
        :param num_ctrlpts: non-zero number of control points
        :type num_ctrlpts: int
        :return: clamped knot vector
        :rtype: Array
        """
        # Number of repetitions at the start and end of the array
        num_repeat = degree
        # Number of knots in the middle
        num_segments = int(num_ctrlpts - (degree + 1))

        return jnp.concatenate(
            (
                jnp.zeros(num_repeat),
                jnp.linspace(0.0, 1.0, num_segments + 2),
                jnp.ones(num_repeat),
            )
        )

    @staticmethod
    def generate_unclamped_knots(degree: int, num_ctrlpts: int):
        """Generates a unclamped knot vector.

        :param degree: non-zero degree of the curve
        :type degree: int
        :param num_ctrlpts: non-zero number of control points
        :type num_ctrlpts: int
        :return: clamped knot vector
        :rtype: Array
        """
        # Should conform the rule: m = n + p + 1
        return jnp.linspace(0.0, 1.0, degree + num_ctrlpts + 1)


    @staticmethod
    def eval_curve_pnt(
        degree: int, ctrl_pnts: jax.Array, basis: jax.Array, span: jax.Array
    ):
        dim = ctrl_pnts.shape[-1]
        if len(ctrl_pnts) < degree + 1:
            raise ValueError("Invalid size of control points for the given degree.")

        ctrl_pnt_slice = jax.lax.dynamic_slice(
            ctrl_pnts, (span - degree, 0), (1 + degree, dim)
        )
        return jnp.sum(ctrl_pnt_slice * jnp.expand_dims(basis, axis=1), axis=0)

    @staticmethod
    def eval_curve(
       curve: BsplineCurve, u: jax.Array
    ):
        knots = (
            BSplineEvaluator.generate_clamped_knots(curve.degree, len(curve.ctrl_pnts))
            if curve.knots is None
            else curve.knots
        )

        span = BSplineEvaluator.find_spans(curve.degree, knots, len(curve.ctrl_pnts), u)
        basis = BSplineEvaluator.basis_functions(curve.degree, knots, span, u)

        return jax.vmap(BSplineEvaluator.eval_curve_pnt, in_axes=(None, None, 1, 0))(
            curve.degree, curve.ctrl_pnts, basis, span
        )

    @staticmethod
    def eval_surface_pnt(
        u_degree: int,
        v_degree: int,
        ctrl_pnts: jax.Array,
        basis_u: jax.Array,
        basis_v: jax.Array,
        span_u: jax.Array,
        span_v: jax.Array,
    ):
        ctrl_pnt_slice = jax.lax.dynamic_slice(
            ctrl_pnts,
            (span_u - u_degree, span_v - v_degree, 0),
            (1 + u_degree, 1 + v_degree, 3),
        )

        eval_prev_pnt = jnp.sum(ctrl_pnt_slice * jnp.expand_dims(basis_v, axis=1), axis=1)
        return jnp.sum(eval_prev_pnt * jnp.expand_dims(basis_u, axis=1), axis=0)

    @staticmethod
    def eval_surface(
        surf: BsplineSurface,
        u: jax.Array,
        v: jax.Array,
    ):

        assert (
            surf.ctrl_pnts.shape[0] >= surf.u_degree + 1
        ), f"Number of curves should be at least {surf.u_degree + 1}"
        assert (
            surf.ctrl_pnts.shape[1] >= surf.v_degree + 1
        ), f"Number of control points should be at least {surf.v_degree + 1}"

        span_u = BSplineEvaluator.find_spans(
            surf.u_degree, surf.u_knots, surf.ctrl_pnts.shape[0], u
        )
        basis_u = BSplineEvaluator.basis_functions(surf.u_degree, surf.u_knots, span_u, u)

        span_v = BSplineEvaluator.find_spans(
            surf.v_degree, surf.v_knots, surf.ctrl_pnts.shape[1], v
        )
        basis_v = BSplineEvaluator.basis_functions(surf.v_degree, surf.v_knots, span_v, v)

        return jax.vmap(
            jax.vmap(
                BSplineEvaluator.eval_surface_pnt,
                in_axes=(None, None, None, None, 1, None, 0),
            ),
                in_axes=(None, None, None, 1, None, 0, None),
        )(surf.u_degree, surf.v_degree, surf.ctrl_pnts, basis_u, basis_v, span_u, span_v)

