/* Result 1: density and gradient (rows nu, eta, a, w1, w2).  Ports ../src/ddm_fast.py.
   Check: c++ -O2 -std=c++11 -DFDDM_TEST ddm_sz.cpp Faddeeva.cc -o check && ./check */
#include "ddm_sz.h"
#include "Faddeeva.hh"
#include <cmath>
#include <complex>
#include <cstring>

using std::complex;

static const double SQ2PI = std::sqrt(2 * M_PI), RT1_2 = std::sqrt(0.5);

/* e^K [Phi(x2) - Phi(x1)] via the Mills ratio, with E_i = e^K e^{-x_i^2/2} (ddm_kernel.pyx). */
static inline double eK_dPhi(double x1, double x2, double K, double E1, double E2) {
    if (x1 >= 0) return 0.5 * (E1 * Faddeeva::erfcx(x1 * RT1_2) - E2 * Faddeeva::erfcx(x2 * RT1_2));
    if (x2 <= 0) return 0.5 * (E2 * Faddeeva::erfcx(-x2 * RT1_2) - E1 * Faddeeva::erfcx(-x1 * RT1_2));
    return std::exp(K) * 0.5 * (std::erfc(-x2 * RT1_2) - std::erfc(-x1 * RT1_2));
}

/* Term j is (p + q w) exp(-A w^2/2 + B w + C); d[r] = {dA, dB, dC, dp, dq} for nu, eta, a. */
struct Term {
    double p, q, B, C, sgn, d[3][5];
};
static inline Term term_j(int j, double t, double nu, double eta, double a, double S) {
    Term T;
    int even = (j % 2 == 0), k = even ? j : j + 1;
    double sj = even ? -1.0 : 1.0;
    T.p = k * a;
    T.q = even ? a : -a;
    T.B = -nu * a / S + sj * k * a * a / t;
    T.C = -nu * nu * t / (2 * S) - (k * a) * (k * a) / (2 * t);
    T.sgn = even ? 1.0 : -1.0;
    const double d[3][5] = {
        {0, -a / S, -nu * t / S, 0, 0},
        {-2 * a * a * eta / (S * S), 2 * nu * a * eta * t / (S * S), nu * nu * eta * t * t / (S * S), 0, 0},
        {2 * a / (t * S), -nu / S + 2 * sj * k * a / t, -(double)k * k * a / t, (double)k, even ? 1.0 : -1.0}};
    std::memcpy(T.d, d, sizeof d);
    return T;
}

/* Small-time series, t/a^2 <= 1 (ddm_fast._small). */
static void small_one(double t, double nu, double eta, double a, double w1, double w2,
                      double tol, double *g, double *dg) {
    double S = 1 + eta * eta * t, A = a * a / (t * S), sA = std::sqrt(A);
    double P = 1 / ((w2 - w1) * std::sqrt(2 * M_PI * t * t * t * S));
    int J = (int)std::ceil(std::sqrt(2 * t * std::log(1 / tol)) / a) + 1;
    if (J < 2) J = 2;
    double sum = 0, dsum[5] = {0, 0, 0, 0, 0};
    for (int j = 0; j < J; j++) {
        Term T = term_j(j, t, nu, eta, a, S);
        double m = T.B / A;
        double E1 = std::exp(T.C + T.B * w1 - A * w1 * w1 / 2);
        double E2 = std::exp(T.C + T.B * w2 - A * w2 * w2 / 2);
        double G = SQ2PI / sA * eK_dPhi((w1 - m) * sA, (w2 - m) * sA, T.C + T.B * T.B / (2 * A), E1, E2);
        sum += T.sgn * ((T.p + T.q * m) * G - (T.q / A) * (E2 - E1));
        for (int r = 0; r < 5; r++) {
            double dA = 0, dB = 0, dC = 0, dp = 0, dq = 0, dw1 = (r == 3), dw2 = (r == 4);
            if (r < 3) { dA = T.d[r][0]; dB = T.d[r][1]; dC = T.d[r][2]; dp = T.d[r][3]; dq = T.d[r][4]; }
            double dm = (dB - m * dA) / A, dK = dC + m * dB - m * m * dA / 2;
            double dG = G * (dK - dA / (2 * A)) + E2 * ((dw2 - dm) + (w2 - m) * dA / (2 * A))
                                                - E1 * ((dw1 - dm) + (w1 - m) * dA / (2 * A));
            double dE1 = E1 * (dC + w1 * dB - w1 * w1 * dA / 2 + (T.B - A * w1) * dw1);
            double dE2 = E2 * (dC + w2 * dB - w2 * w2 * dA / 2 + (T.B - A * w2) * dw2);
            dsum[r] += T.sgn * ((dp + dq * m + T.q * dm) * G + (T.p + T.q * m) * dG
                                - (dq / A - T.q * dA / (A * A)) * (E2 - E1) - (T.q / A) * (dE2 - dE1));
        }
    }
    *g = P * sum;
    double dlogP[5] = {0, -eta * t / S, 0, 1 / (w2 - w1), -1 / (w2 - w1)};
    for (int r = 0; r < 5; r++) dg[r] = P * dsum[r] + dlogP[r] * *g;
}

/* Fixed-start small-time density at w (Blurton et al. Eq. 1): f, df/d(nu, eta, a), df/dw. */
static void small_pt(double t, double nu, double eta, double a, double w, double tol,
                     double *f, double *df3, double *dfw) {
    double S = 1 + eta * eta * t, A = a * a / (t * S), P = 1 / std::sqrt(2 * M_PI * t * t * t * S);
    int J = (int)std::ceil(std::sqrt(2 * t * std::log(1 / tol)) / a) + 1;
    if (J < 2) J = 2;
    double sum = 0, dsum[3] = {0, 0, 0}, sw = 0;
    for (int j = 0; j < J; j++) {
        Term T = term_j(j, t, nu, eta, a, S);
        double E = std::exp(T.C + T.B * w - A * w * w / 2), V = (T.p + T.q * w) * E;
        sum += T.sgn * V;
        for (int r = 0; r < 3; r++) {
            const double *d = T.d[r];
            dsum[r] += T.sgn * ((d[3] + w * d[4]) * E + V * (d[2] + w * d[1] - w * w * d[0] / 2));
        }
        sw += T.sgn * (T.q + (T.p + T.q * w) * (T.B - A * w)) * E;
    }
    *f = P * sum;
    for (int r = 0; r < 3; r++) df3[r] = P * dsum[r];
    df3[1] -= eta * t / S * *f;
    *dfw = P * sw;
}

static inline int large_K(double t, double a, double tol) {
    return (int)std::ceil(a * std::sqrt(2 * std::log(1 / tol) / (M_PI * M_PI * t))) + 1;
}
/* rows nu, eta, a from the mu- and kappa-derivatives X1, X2. */
static inline void lchain(complex<double> X1, complex<double> X2, double nu, double eta, double a,
                          double S, double t, complex<double> *out) {
    out[0] = X1 * (-a / S);
    out[1] = X1 * (2 * nu * a * eta * t / (S * S)) + X2 * (eta * a * a / (S * S));
    out[2] = X1 * (-nu / S) + X2 * (eta * eta * a / S);
}

/* Large-time series, t/a^2 >= 1.  Iw = int_{w1}^{w2} exp(kap w^2 + mu_k w) dw. */
static void large_one(double t, double nu, double eta, double a, double w1, double w2,
                      double tol, double *g, double *dg) {
    double S = 1 + eta * eta * t, kap = eta * eta * a * a / (2 * S), lam = -nu * a / S;
    double P = M_PI / (a * a) / std::sqrt(S) * std::exp(-nu * nu * t / (2 * S)) / (w2 - w1);
    int K = large_K(t, a, tol);
    double sum = 0, dsum[5] = {0, 0, 0, 0, 0};
    for (int k = 1; k <= K; k++) {
        complex<double> mu(lam, M_PI * k);
        complex<double> E1 = std::exp(kap * w1 * w1 + mu * w1), E2 = std::exp(kap * w2 * w2 + mu * w2);
        complex<double> Iw, dmu, dkap;
        if (kap < 0.1) {                        /* Taylor in kappa; Pn = int w^n e^{mu w} dw */
            complex<double> e1 = std::exp(mu * w1), e2 = std::exp(mu * w2), Pn[23], T[3] = {0, 0, 0};
            Pn[0] = (e2 - e1) / mu;
            double p1 = 1, p2 = 1;
            for (int n = 1; n < 23; n++) {
                p1 *= w1; p2 *= w2;
                Pn[n] = ((p2 * e2 - p1 * e1) - (double)n * Pn[n - 1]) / mu;
            }
            double kp = 1, fac = 1;
            for (int m = 0; m < 11; m++) {
                if (m) { kp *= kap; fac *= m; }
                for (int i = 0; i < 3; i++) T[i] += kp / fac * Pn[2 * m + i];
            }
            Iw = T[0]; dmu = T[1]; dkap = T[2];
        } else {
            double sk = std::sqrt(kap);
            Iw = complex<double>(0, -1) * std::sqrt(M_PI) / (2 * sk)
                 * (E2 * Faddeeva::w(sk * w2 + mu / (2 * sk)) - E1 * Faddeeva::w(sk * w1 + mu / (2 * sk)));
            dmu = ((E2 - E1) - mu * Iw) / (2 * kap);
            dkap = ((w2 * E2 - w1 * E1) - Iw - mu * dmu) / (2 * kap);
        }
        double damp = k * std::exp(-(double)k * k * M_PI * M_PI * t / (2 * a * a));
        sum += damp * Iw.imag();
        complex<double> dI[5];
        lchain(dmu, dkap, nu, eta, a, S, t, dI);
        dI[3] = -E1; dI[4] = E2;
        for (int r = 0; r < 5; r++) dsum[r] += damp * dI[r].imag();
        dsum[2] += damp * (double)k * k * M_PI * M_PI * t / (a * a * a) * Iw.imag();
    }
    *g = P * sum;
    double dlogP[5] = {-nu * t / S, -eta * t / S + nu * nu * eta * t * t / (S * S), -2 / a,
                       1 / (w2 - w1), -1 / (w2 - w1)};
    for (int r = 0; r < 5; r++) dg[r] = P * dsum[r] + dlogP[r] * *g;
}

/* Fixed-start large-time density at w: f, df/d(nu, eta, a), df/dw. */
static void large_pt(double t, double nu, double eta, double a, double w, double tol,
                     double *f, double *df3, double *dfw) {
    double S = 1 + eta * eta * t, kap = eta * eta * a * a / (2 * S), lam = -nu * a / S;
    double P = M_PI / (a * a) / std::sqrt(S) * std::exp(-nu * nu * t / (2 * S));
    int K = large_K(t, a, tol);
    double sum = 0, dsum[3] = {0, 0, 0}, sw = 0;
    for (int k = 1; k <= K; k++) {
        complex<double> mu(lam, M_PI * k), e = std::exp(kap * w * w + mu * w), row[3];
        double damp = k * std::exp(-(double)k * k * M_PI * M_PI * t / (2 * a * a));
        sum += damp * e.imag();
        lchain(w * e, w * w * e, nu, eta, a, S, t, row);
        for (int r = 0; r < 3; r++) dsum[r] += damp * row[r].imag();
        dsum[2] += damp * (double)k * k * M_PI * M_PI * t / (a * a * a) * e.imag();
        sw += damp * ((2 * kap * w + mu) * e).imag();
    }
    *f = P * sum;
    double dlogP[3] = {-nu * t / S, -eta * t / S + nu * nu * eta * t * t / (S * S), -2 / a};
    for (int r = 0; r < 3; r++) df3[r] = P * dsum[r] + dlogP[r] * *f;
    *dfw = P * sw;
}

static const double GX5[5] = {-0.9061798459386640, -0.5384693101056831, 0.0,
                              0.5384693101056831, 0.9061798459386640};
static const double GW5[5] = {0.1184634425280945, 0.2393143352496832, 0.2844444444444444,
                              0.2393143352496832, 0.1184634425280945};

/* w2 - w1 < 1e-2: F(w2) - F(w1) loses digits like 1/(w2 - w1); average 5 fixed-start points. */
static void avg_one(double t, double nu, double eta, double a, double w1, double w2,
                    double tol, double *g, double *dg) {
    double wb = (w1 + w2) / 2, D = w2 - w1;
    int large = t / (a * a) > 1.0;
    *g = 0;
    for (int r = 0; r < 5; r++) dg[r] = 0;
    for (int n = 0; n < 5; n++) {
        double f, d3[3], fw, om = GW5[n];      /* GW5 sums to 1 */
        if (large) large_pt(t, nu, eta, a, wb + D * GX5[n] / 2, tol, &f, d3, &fw);
        else       small_pt(t, nu, eta, a, wb + D * GX5[n] / 2, tol, &f, d3, &fw);
        *g += om * f;
        for (int r = 0; r < 3; r++) dg[r] += om * d3[r];
        dg[3] += om * fw * (1 - GX5[n]) / 2;   /* d/dw1, d/dw2 through (wbar, Delta) */
        dg[4] += om * fw * (1 + GX5[n]) / 2;
    }
}

static inline int params_ok(double eta, double a, double w1, double w2, double tol) {
    return a > 0 && eta >= 0 && 0 < w1 && w1 <= w2 && w2 < 1 && tol > 0 && tol < 1;
}

static void g_one(double t, double nu, double eta, double a, double w1, double w2, double tol,
                  double *g, double *dg) {
    if (!(t > 0)) {
        *g = 0;
        for (int r = 0; r < 5; r++) dg[r] = 0;
        return;
    }
    int large = t / (a * a) > 1.0;              /* small-/large-time switch at t/a^2 = 1 */
    if (w2 - w1 < 1e-2) avg_one(t, nu, eta, a, w1, w2, tol, g, dg);
    else if (large) large_one(t, nu, eta, a, w1, w2, tol, g, dg);
    else small_one(t, nu, eta, a, w1, w2, tol, g, dg);
}

int ddm_sz(const double *t, int n, double nu, double eta, double a, double w1, double w2,
           double tol, double *g, double *dg) {
    if (!params_ok(eta, a, w1, w2, tol)) return 1;
    for (int i = 0; i < n; i++) g_one(t[i], nu, eta, a, w1, w2, tol, g + i, dg + 5 * i);
    return 0;
}

/* 48-node Gauss-Legendre; 32 lose digits for w1 < 0.2. */
static const double GX48[48] = {
    -0.9987710072524261, -0.9935301722663508, -0.9841245837228269, -0.9705915925462473,
    -0.9529877031604308, -0.9313866907065543, -0.9058791367155696, -0.8765720202742479,
    -0.8435882616243935, -0.8070662040294426, -0.7671590325157404, -0.7240341309238146,
    -0.6778723796326639, -0.6288673967765136, -0.5772247260839727, -0.5231609747222330,
    -0.4669029047509584, -0.4086864819907167, -0.3487558862921607, -0.2873624873554555,
    -0.2247637903946890, -0.1612223560688917, -0.0970046992094627, -0.0323801709628693,
     0.0323801709628693,  0.0970046992094627,  0.1612223560688917,  0.2247637903946890,
     0.2873624873554555,  0.3487558862921607,  0.4086864819907167,  0.4669029047509584,
     0.5231609747222330,  0.5772247260839727,  0.6288673967765136,  0.6778723796326639,
     0.7240341309238146,  0.7671590325157404,  0.8070662040294426,  0.8435882616243935,
     0.8765720202742479,  0.9058791367155696,  0.9313866907065543,  0.9529877031604308,
     0.9705915925462473,  0.9841245837228269,  0.9935301722663508,  0.9987710072524261};
static const double GW48[48] = {
    0.0031533460523092, 0.0073275539012762, 0.0114772345792350, 0.0155793157229431,
    0.0196161604573557, 0.0235707608393240, 0.0274265097083569, 0.0311672278327984,
    0.0347772225647706, 0.0382413510658306, 0.0415450829434645, 0.0446745608566941,
    0.0476166584924903, 0.0503590355538543, 0.0528901894851935, 0.0551995036999841,
    0.0572772921004029, 0.0591148396983954, 0.0607044391658936, 0.0620394231598924,
    0.0631141922862538, 0.0639242385846479, 0.0644661644359498, 0.0647376968126839,
    0.0647376968126839, 0.0644661644359498, 0.0639242385846479, 0.0631141922862538,
    0.0620394231598924, 0.0607044391658936, 0.0591148396983954, 0.0572772921004029,
    0.0551995036999841, 0.0528901894851935, 0.0503590355538543, 0.0476166584924903,
    0.0446745608566941, 0.0415450829434645, 0.0382413510658306, 0.0347772225647706,
    0.0311672278327984, 0.0274265097083569, 0.0235707608393240, 0.0196161604573557,
    0.0155793157229431, 0.0114772345792350, 0.0073275539012762, 0.0031533460523092};

int ddm_sz7(const double *t, int n, double nu, double eta, double a, double w1, double w2,
            double t0, double st0, double tol, double *f, double *df) {
    if (!params_ok(eta, a, w1, w2, tol) || st0 < 0) return 1;
    for (int i = 0; i < n; i++) {
        double *dfi = df + 7 * i, hi = t[i] - t0, g_hi, g_lo, junk[5];
        for (int r = 0; r < 7; r++) dfi[r] = 0;
        f[i] = 0;
        if (hi <= 0) continue;
        if (st0 == 0) {                         /* ponytail: g'(hi) by difference, ~8 digits; derive dg/dt if it matters */
            double h = 1e-5 * (hi > 1 ? hi : 1), up, dn, d[5];
            g_one(hi, nu, eta, a, w1, w2, tol, f + i, dfi);
            g_one(hi + h, nu, eta, a, w1, w2, tol, &up, d);
            g_one(hi - h, nu, eta, a, w1, w2, tol, &dn, d);
            dfi[5] = -(up - dn) / (2 * h);      /* d/dt0 = -g'(hi) */
            dfi[6] = dfi[5] / 2;                /* d/dst0 = -g'(hi)/2 */
            continue;
        }
        double lo = hi - st0 > 0 ? hi - st0 : 0;    /* g(0) = 0 covers the clamped window */
        for (int k = 0; k < 48; k++) {          /* cubic map clusters nodes at the lower limit */
            double s = (GX48[k] + 1) / 2, jac = (hi - lo) * 3 * s * s / 2, g, dg[5];
            g_one(lo + (hi - lo) * s * s * s, nu, eta, a, w1, w2, tol, &g, dg);
            f[i] += GW48[k] * jac * g;
            for (int r = 0; r < 5; r++) dfi[r] += GW48[k] * jac * dg[r];
        }
        f[i] /= st0;
        for (int r = 0; r < 5; r++) dfi[r] /= st0;
        g_one(hi, nu, eta, a, w1, w2, tol, &g_hi, junk);
        g_one(lo, nu, eta, a, w1, w2, tol, &g_lo, junk);
        dfi[5] = (g_lo - g_hi) / st0;
        dfi[6] = (g_lo - f[i]) / st0;
    }
    return 0;
}

/* (w1, w2) -> (w, sz) on the two starting-point rows, in place. */
static inline void to_w_sz(double *d, int stride, int n) {
    for (int i = 0; i < n; i++) {
        double *r = d + stride * i, dw1 = r[3], dw2 = r[4];
        r[3] = dw1 + dw2;
        r[4] = (dw2 - dw1) / 2;
    }
}

int ddm_sz_w(const double *t, int n, double nu, double eta, double a, double w, double sz,
             double tol, double *g, double *dg) {
    int bad = ddm_sz(t, n, nu, eta, a, w - sz / 2, w + sz / 2, tol, g, dg);
    if (!bad) to_w_sz(dg, 5, n);
    return bad;
}

int ddm_sz7_w(const double *t, int n, double nu, double eta, double a, double w, double sz,
              double t0, double st0, double tol, double *f, double *df) {
    int bad = ddm_sz7(t, n, nu, eta, a, w - sz / 2, w + sz / 2, t0, st0, tol, f, df);
    if (!bad) to_w_sz(df, 7, n);
    return bad;
}

#ifdef FDDM_TEST
/* vs ../../data/wienr_grad.csv (WienR, 400 sets) and central differences. */
#include <cstdio>
#include <cstdlib>
#include <algorithm>

static double dens(double t, double nu, double eta, double a, double w1, double w2) {
    double g, dg[5];
    if (ddm_sz(&t, 1, nu, eta, a, w1, w2, 1e-12, &g, dg)) { std::printf("bad params\n"); exit(1); }
    return g;
}

int main() {
    const char *nm[6] = {"g", "d/dnu", "d/deta", "d/da", "d/dw", "d/dsw"};
    for (int f = 0; f < 2; f++) {           /* fixed 400-set grid, then the 2030 published sets */
        const char *path = f ? "../../data/tran_grid.csv" : "../../data/wienr_grad.csv";
        FILE *fp = std::fopen(path, "r");
        if (!fp) { std::printf("run from fddm-fpt/ (needs %s)\n", path); return 1; }
        char line[1024];
        if (!std::fgets(line, sizeof line, fp)) return 1;              /* header */
        double worst[6] = {0, 0, 0, 0, 0, 0};
        int rows = 0;
        while (std::fgets(line, sizeof line, fp)) {
            double t, a, v, w, sv, sw, ref[6];
            if (std::sscanf(line, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf", &t, &a, &v, &w,
                            &sv, &sw, ref, ref + 1, ref + 2, ref + 3, ref + 4, ref + 5) != 12) continue;
            if (ref[0] < 1e-10) continue;                              /* WienR's own error dominates */
            double g, dg[5];
            ddm_sz_w(&t, 1, v, sv, a, w, sw, 1e-12, &g, dg);   /* WienR's (w, sw) */
            double ours[6] = {g, dg[0], dg[1], dg[2], dg[3], dg[4]};
            for (int c = 0; c < 6; c++)
                worst[c] = std::max(worst[c], std::fabs(ours[c] - ref[c]) / std::max(std::fabs(ref[c]), g));
            rows++;
        }
        std::fclose(fp);
        std::printf("vs WienR, %d sets:", rows);
        for (int c = 0; c < 6; c++) std::printf("  %s %.1e", nm[c], worst[c]);
        std::printf("\n");
        if (rows < 300) { std::printf("FAIL: %s parsed only %d rows\n", path, rows); return 1; }
        for (int c = 0; c < 6; c++) if (!(worst[c] < 1e-8)) { std::printf("FAIL: %s\n", nm[c]); return 1; }
    }

    /* gradient vs central differences in each branch, including the small-Delta average */
    double h = 1e-6, wmax = 0;
    for (int b = 0; b < 3; b++) {
        double t = b == 0 ? 0.5 : 2.5, W = b == 2 ? 0.002 : 0.3;       /* small-time, large-time, average */
        double p[5] = {0.7, 1.1, 1.2, 0.5 - W / 2, 0.5 + W / 2}, g, dg[5];
        ddm_sz(&t, 1, p[0], p[1], p[2], p[3], p[4], 1e-12, &g, dg);
        for (int r = 0; r < 5; r++) {
            double q[5]; std::copy(p, p + 5, q);
            q[r] = p[r] + h; double up = dens(t, q[0], q[1], q[2], q[3], q[4]);
            q[r] = p[r] - h; double dn = dens(t, q[0], q[1], q[2], q[3], q[4]);
            double fd = (up - dn) / (2 * h);
            wmax = std::max(wmax, std::fabs(dg[r] - fd) / std::max(std::fabs(fd), g));
        }
    }
    std::printf("gradient vs finite differences, three branches: max rel %.1e\n", wmax);
    if (!(wmax < 1e-6)) { std::printf("FAIL: finite differences\n"); return 1; }

    /* seven parameters vs finite differences, full and clamped window */
    double w7 = 0;
    for (int b = 0; b < 2; b++) {
        double p[7] = {0.7, 1.1, 1.2, 0.35, 0.65, 0.2, 0.1}, t = b ? 0.25 : 0.9, f, df[7];
        ddm_sz7(&t, 1, p[0], p[1], p[2], p[3], p[4], p[5], p[6], 1e-12, &f, df);
        for (int r = 0; r < 7; r++) {
            double q[7], up, dn, junk[7];
            std::copy(p, p + 7, q);
            q[r] = p[r] + h; ddm_sz7(&t, 1, q[0], q[1], q[2], q[3], q[4], q[5], q[6], 1e-12, &up, junk);
            q[r] = p[r] - h; ddm_sz7(&t, 1, q[0], q[1], q[2], q[3], q[4], q[5], q[6], 1e-12, &dn, junk);
            double fd = (up - dn) / (2 * h);
            w7 = std::max(w7, std::fabs(df[r] - fd) / std::max(std::fabs(fd), f));
        }
    }
    std::printf("seven-parameter gradient vs finite differences: max rel %.1e\n", w7);
    if (!(w7 < 1e-6)) { std::printf("FAIL: f7 finite differences\n"); return 1; }
    std::printf("DDM_SZ CHECK PASS\n");
    return 0;
}
#endif
