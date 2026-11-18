"""prrax's mantis algorithms, traced here from ``mantis.lib`` (nothing added to mantis itself).

* ``sphere_frames``: ``[t, R column-major]`` of each sphere-carrying frame (the collision check's FK).
* ``tsr_step(i)``: one damped-LM step of a TSR constraint on end-effector frame ``i``, FK + Jacobian
  + hinge + 6x6 Cholesky fused: ``e2 = |d|^2``, ``g = Jx^T (Jx Jx^T + lam I)^-1 d`` (q <- q - alpha g),
  the same step as prrtc_constraint.cuh's GLASS / GRiM path. The constraint is a per-call input,
  laid out as the planner's packed constraint from CON_RTE_INV: rTe^-1 [wxyz xyz], wTr^-1
  [wxyz xyz], lo [6], hi [6]. The Jacobian was checked against mantis's forward AD (< 1e-7 for
  theta in [0, 3]).
"""
from pyroffi import _grim

_grim.ensure_importable()   # puts GRiM's vendored mantis on sys.path

import mantis  # noqa: E402
from mantis import lib  # noqa: E402

N_CON = 26   # rTe^-1 [wxyz xyz], wTr^-1 [wxyz xyz], lo [6], hi [6]


def _mm(A, B):
    return [[sum((A[i][k] * B[k][j] for k in range(1, 3)), A[i][0] * B[0][j]) for j in range(3)] for i in range(3)]


def _mv(A, v):
    return [A[i][0] * v[0] + A[i][1] * v[1] + A[i][2] * v[2] for i in range(3)]


def _cross(a, b):
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


def tsr_terms(robot, q, con, frame):
    """d (hinged TSR error, 6) and Jx = dx/dq (6 x n_q rows) of constraint ``con`` on ``frame``."""
    Xe, J = lib.jacobian(robot, lib.poses(robot, q), frame)
    e = [Xe[k] for k in range(12)]
    Ree = [[e[3 * c + r] for c in range(3)] for r in range(3)]   # R = E^T
    pee = [e[9 + i] for i in range(3)]
    Rc, pc = lib.quat_matrix(*con[0:4]), con[4:7]
    Rw, pw = lib.quat_matrix(*con[7:11]), con[11:14]
    off = _mv(Ree, pc)                                           # p_obj - p_ee
    pobj = [pee[i] + off[i] for i in range(3)]
    Rr = _mm(Rw, _mm(Ree, Rc))
    t = [v + pw[i] for i, v in enumerate(_mv(Rw, pobj))]
    phi = lib.so3_log(Rr)
    x = t + phi
    lo, hi = con[14:20], con[20:26]
    d = [(x[i] - hi[i]).maximum(0.0) + (x[i] - lo[i]).minimum(0.0) for i in range(6)]
    # Jl^-1(phi) = I - [phi]x / 2 + c [phi]x^2, c = 1/th^2 - (1 + cos) / (2 th sin): exact form
    # gated off below th ~ 0.3 in favour of its Taylor series (the exact form cancels there).
    th2 = phi[0] * phi[0] + phi[1] * phi[1] + phi[2] * phi[2]
    th = th2.sqrt()
    s, c0 = th.sin(), th.cos()
    exact = (1.0 - th * s / (2.0 * (1.0 - c0) + 1e-30)) / (th2 + 1e-30)
    w4 = th2 * th2 * th2 * th2
    w = w4 / (w4 + 0.3 ** 8)
    cc = w * exact + (1.0 - w) * (1.0 / 12.0 + th2 / 720.0)
    n = robot.n_q
    Jx = []
    Jv = [[J[r][k] for r in range(3)] for k in range(n)]        # per column
    Jw = [[J[3 + r][k] for r in range(3)] for k in range(n)]
    colt, colr = [], []
    for k in range(n):
        colt.append(_mv(Rw, [Jv[k][i] + v for i, v in enumerate(_cross(Jw[k], off))]))
        a = _mv(Rw, Jw[k])
        pa = _cross(phi, a)
        ppa = _cross(phi, pa)
        colr.append([a[i] - 0.5 * pa[i] + cc * ppa[i] for i in range(3)])
    Jx = [[colt[k][r] for k in range(n)] for r in range(3)] + [[colr[k][r] for k in range(n)] for r in range(3)]
    return d, Jx, x


def sphere_frames():
    @mantis.algorithm(inputs=("q",), outputs=("poses",), name="prrax_sphere_frames", register=False)
    def prrax_sphere_frames(robot, q, frames=()):
        X0 = lib.poses(robot, q)
        return [lib.pose_out(lib.frame(robot, X0, f)) for f in frames]
    return prrax_sphere_frames


def tsr_step(i: int):
    @mantis.algorithm(inputs=("q", ("con", lambda o: N_CON), ("lam", lambda o: 1)), outputs=("e2", "g"),
                      name=f"prrax_tsr_step_{i}", register=False)
    def prrax_tsr_step(robot, q, con, lam, frame=None):
        """e2 = |d|^2 and the step g = Jx^T (Jx Jx^T + lam I)^-1 d (q <- q - alpha g)."""
        d, Jx, _ = tsr_terms(robot, q, con, frame)
        n = robot.n_q
        A = [[lib.sum_tree([Jx[i][k] * Jx[j][k] for k in range(n)], None) + (lam[0] if i == j else 0.0)
              for j in range(6)] for i in range(6)]
        y = lib.cholesky_solve(A, d)
        g = [lib.sum_tree([Jx[r][k] * y[r] for r in range(6)], None) for k in range(n)]
        e2 = lib.sum_tree([v * v for v in d], None)
        return e2, g
    return prrax_tsr_step



def _dist(a, b, ra, rb):
    v = [a[i] - b[i] for i in range(3)]
    return (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt() - (ra + rb)


def edge_margin():
    """One edge waypoint's whole validation step, traced: ``q = q0 + t * delta``, FK, every fine
    sphere centre, and the minimum signed distance over (sphere x obstacle) and the self pairs;
    ``margin <= 0`` is a collision. Obstacles are per-call inputs, their counts compile-time
    (``n_spheres``, ``n_capsules``, ``n_boxes``, ``n_halfspaces``: pad a smaller world with
    obstacles out of reach). No early exit: every distance is computed."""
    kinds = (("spheres", 4), ("capsules", 7), ("boxes", 15), ("halfspaces", 6))
    world = tuple((k, lambda o, k=k, w=w: w * o.get(f"n_{k}", 0)) for k, w in kinds)

    @mantis.algorithm(inputs=("q", "delta", ("t", lambda o: 1)) + world, outputs=("margin",),
                      name="prrax_edge_margin", register=False)
    def prrax_edge_margin(robot, q, delta, t, spheres, capsules, boxes, halfspaces, rows=(), pairs=(),
                          n_spheres=0, n_capsules=0, n_boxes=0, n_halfspaces=0):
        """``rows``: ``(frame or None for the base, (x, y, z), radius)`` fine spheres; ``pairs``:
        self pairs of rows."""
        qt = [q[i] + t[0] * delta[i] for i in range(len(q))]
        moving = [(f, c) for f, c, _ in rows if f is not None]
        cw = lib.sphere_centres(robot, lib.poses(robot, qt), moving) if moving else []
        c, k = [], 0
        for f, centre, _ in rows:          # frame None: a base sphere, fixed in the world
            if f is None:
                c.append([lib.Scalar.const(robot.t, float(x)) for x in centre])
            else:
                c.append(cw[3 * k:3 * k + 3])
                k += 1
        r = [float(x) for _, _, x in rows]
        obs = dict(spheres=spheres, capsules=capsules, boxes=boxes, halfspaces=halfspaces)
        terms = [_dist(c[i], c[j], r[i], r[j]) for i, j in pairs]
        for kind, w in kinds:
            for o in range(len(obs[kind]) // w):
                ob = obs[kind][w * o:w * o + w]
                terms += [lib.obstacle_distance(kind, c[s], r[s], ob) for s in range(len(rows))]
        return lib.min_tree(terms)
    return prrax_edge_margin


def coarse_margins():
    """The coarse pre-pass of ``binary_coarse``, traced: FK of the coarse frames, then per coarse
    sphere its minimum signed distance over every obstacle, then per coarse self pair its distance.
    Branch-free; the fine checks behind a coarse margin <= 0 stay runtime loops. Inputs as
    :func:`edge_margin` without the interpolation (``q`` is the waypoint)."""
    kinds = (("spheres", 4), ("capsules", 7), ("boxes", 15), ("halfspaces", 6))
    world = tuple((k, lambda o, k=k, w=w: w * o.get(f"n_{k}", 0)) for k, w in kinds)

    @mantis.algorithm(inputs=("q",) + world, outputs=("margins",), name="prrax_coarse_margins", register=False)
    def prrax_coarse_margins(robot, q, spheres, capsules, boxes, halfspaces, rows=(), pairs=(),
                             n_spheres=0, n_capsules=0, n_boxes=0, n_halfspaces=0):
        moving = [(f, c) for f, c, _ in rows if f is not None]
        cw = lib.sphere_centres(robot, lib.poses(robot, list(q)), moving) if moving else []
        c, k = [], 0
        for f, centre, _ in rows:
            if f is None:
                c.append([lib.Scalar.const(robot.t, float(x)) for x in centre])
            else:
                c.append(cw[3 * k:3 * k + 3])
                k += 1
        r = [float(x) for _, _, x in rows]
        obs = dict(spheres=spheres, capsules=capsules, boxes=boxes, halfspaces=halfspaces)
        per = [[] for _ in rows]
        for kind, w in kinds:
            for o in range(len(obs[kind]) // w):
                ob = obs[kind][w * o:w * o + w]
                for s in range(len(rows)):
                    per[s].append(lib.obstacle_distance(kind, c[s], r[s], ob))
        far = lib.Scalar.const(robot.t, 1e30)
        return [lib.min_tree(t) if t else far for t in per] + [_dist(c[i], c[j], r[i], r[j]) for i, j in pairs]
    return prrax_coarse_margins
