// FLASK (Duong, Ramsey, Kingston, Thomason, Kavraki; T-RO 2026, arXiv 2603.16059): kinodynamic
// edges for a fully actuated arm by differential flatness. The flat state is (q, qd) and the
// pseudo-control w = qdd (a double integrator per joint). Between two states the fixed-T LQMT
// BVP is solved by the cubic Hermite (minimum control effort), with closed-form cost
//     C(T) = int_0^T |qdd|^2 dt + rho T
//          = 12|d|^2 / T^3 - 12 (v0 + v1).d / T^2 + 4 (|v0|^2 + v0.v1 + |v1|^2) / T + rho T,
// d = q1 - q0 (Eq. 27-28), and the edge's duration is the minimizer T* (a positive root of the
// quartic dC/dT = 0). Validation (Alg. 3) samples the cubic: joint, velocity, acceleration and
// torque limits per sample (torque by the includer's per-thread inverse dynamics), positions
// through prrax's collision check.
#pragma once

namespace flask {

struct Limits {
    const float *lo, *hi;            // joint limits [n]
    const float *vmax, *amax, *tmax; // |qd|, |qdd|, |tau| limits [n] (tmax nullptr: no torque check)
    float rho;                       // time weight of C(T)
};

// Cubic Hermite q(t) = q0 + v0 t + c2 t^2 + c3 t^3 through (q0, v0) at 0 and (q1, v1) at T.
__device__ __forceinline__ void coeffs(const float* q0, const float* v0, const float* q1, const float* v1,
                                       int n, float T, float* c2, float* c3) {
    const float iT = 1.f / T, iT2 = iT * iT;
    for (int k = 0; k < n; ++k) {
        const float h = q1[k] - q0[k];
        c2[k] = (3.f * h - (2.f * v0[k] + v1[k]) * T) * iT2;
        c3[k] = (-2.f * h + (v0[k] + v1[k]) * T) * iT2 * iT;
    }
}

__device__ __forceinline__ void eval(const float* q0, const float* v0, const float* c2, const float* c3, int n,
                                     float t, float* q, float* v, float* a) {
    for (int k = 0; k < n; ++k) {
        q[k] = q0[k] + t * (v0[k] + t * (c2[k] + t * c3[k]));
        v[k] = v0[k] + t * (2.f * c2[k] + 3.f * t * c3[k]);
        a[k] = 2.f * c2[k] + 6.f * t * c3[k];
    }
}

struct Terms { float D, p, s; };   // |d|^2, (v0 + v1).d, |v0|^2 + v0.v1 + |v1|^2

__device__ __forceinline__ Terms terms(const float* q0, const float* v0, const float* q1, const float* v1, int n) {
    Terms t{0.f, 0.f, 0.f};
    for (int k = 0; k < n; ++k) {
        const float d = q1[k] - q0[k];
        t.D += d * d;
        t.p += (v0[k] + v1[k]) * d;
        t.s += v0[k] * v0[k] + v0[k] * v1[k] + v1[k] * v1[k];
    }
    return t;
}

__device__ __forceinline__ float cost(const Terms& m, float T, float rho) {
    const float iT = 1.f / T;
    return iT * (4.f * m.s + iT * (-12.f * m.p + iT * 12.f * m.D)) + rho * T;
}

// T* = argmin_{T > 0} C(T), and C(T*) in *c. C -> +inf at both ends (unless the states
// coincide), but is not convex in general: a log-spaced scan over [1e-3, 1e3] s brackets the
// global minimum, golden-section search refines it.
__device__ inline float tstar(const Terms& m, float rho, float* c) {
    if (m.D < 1e-12f && m.s < 1e-12f) { *c = 0.f; return 0.f; }
    constexpr int NS = 48;
    const float l0 = -3.f * 2.302585f, dl = 6.f * 2.302585f / (NS - 1);
    int best = 0;
    float cb = FLT_MAX;
    for (int i = 0; i < NS; ++i) {
        const float ci = cost(m, expf(l0 + i * dl), rho);
        if (ci < cb) { cb = ci; best = i; }
    }
    float a = l0 + max(best - 1, 0) * dl, b = l0 + min(best + 1, NS - 1) * dl;
    constexpr float g = 0.381966f;
    float x1 = a + g * (b - a), x2 = b - g * (b - a);
    float f1 = cost(m, expf(x1), rho), f2 = cost(m, expf(x2), rho);
    for (int it = 0; it < 40; ++it) {
        if (f1 < f2) { b = x2; x2 = x1; f2 = f1; x1 = a + g * (b - a); f1 = cost(m, expf(x1), rho); }
        else         { a = x1; x1 = x2; f1 = f2; x2 = b - g * (b - a); f2 = cost(m, expf(x2), rho); }
    }
    const float T = expf(0.5f * (a + b));
    *c = cost(m, T, rho);
    return T;
}

// Exact joint / velocity / acceleration limits of the cubic on [0, T]: qdd is linear (extremes at
// the ends), qd quadratic (ends + its vertex), q cubic (ends + the roots of qd).
__device__ inline bool bounds_violated(const float* q0, const float* v0, const float* c2, const float* c3,
                                       int n, float T, const Limits& L) {
    for (int k = 0; k < n; ++k) {
        const float A = 3.f * c3[k], B = 2.f * c2[k], C = v0[k];             // qd(t) = A t^2 + B t + C
        auto q = [&](float t) { return q0[k] + t * (C + t * (c2[k] + t * c3[k])); };
        auto v = [&](float t) { return C + t * (B + t * A); };
        if (fabsf(B) > L.amax[k] || fabsf(B + 2.f * A * T) > L.amax[k]) return true;
        float tv[3] = {0.f, T, fabsf(A) > 1e-12f ? -B / (2.f * A) : -1.f};
        for (float t : tv) if (t >= 0.f && t <= T && fabsf(v(t)) > L.vmax[k]) return true;
        float tq[3] = {T, -1.f, -1.f};
        if (fabsf(A) > 1e-12f) {
            const float disc = B * B - 4.f * A * C;
            if (disc >= 0.f) { const float r = sqrtf(disc); tq[1] = (-B - r) / (2.f * A); tq[2] = (-B + r) / (2.f * A); }
        } else if (fabsf(B) > 1e-12f) tq[1] = -C / B;
        for (float t : tq) if (t >= 0.f && t <= T && (q(t) < L.lo[k] || q(t) > L.hi[k])) return true;
    }
    return false;
}

}  // namespace flask
