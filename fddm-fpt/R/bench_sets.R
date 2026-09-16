# The benchmark's parameter sets: 100 draws from the informative priors of Tran, van Maanen,
# Heathcote & Matzke (2021, Front. Psychol., Table 1), truncated at that review's empirical
# bounds.  All seven parameters come from the paper, including t0 and st0.  Both bench.R and
# bench.py read the result, so the two languages run on identical sets.
#   Rscript bench_sets.R  ->  bench_sets.csv
#
# Columns are neutral: a full separation, w and sw relative, t0 the LOWER edge of the
# non-decision window of width st0.  bench.R uses them as rtdists' (a, z = w a, sz = sw a, t0,
# st0); bench.py as HSSM's (a/2, z = w, sz = sw, t = t0 + st0/2, st = st0).
NSET <- as.integer(Sys.getenv("NSET", 100))
set.seed(20260916)

# Table 1: family, location, scale, df, and the empirical [min, max] the review reports
rtnorm <- function(n, m, s, lo, hi) { x <- rnorm(20*n, m, s); head(x[x > lo & x < hi], n) }
rtt    <- function(n, m, s, df, lo, hi) { x <- m + s*rt(200*n, df); head(x[x > lo & x < hi], n) }

draw <- function(n) data.frame(
  a   = pmin(pmax(rgamma(n, 11.69, scale = 0.12), 0.11), 7.47),   # gamma(11.69, 0.12), [0.11, 7.47]
  v   = sample(c(-1, 1), n, TRUE) * rtnorm(n, 1.76, 1.51, 0.01, 18.51),
  w   = rtt(n, 0.50, 0.05, 1.85, 0.04, 0.96),                     # z_r
  sv  = rtnorm(n, 1.36, 0.69, 0.00, 3.45),
  sw  = rtnorm(n, 0.33, 0.22, 0.01, 0.85),                        # s_z_r
  t0  = rtt(n, 0.44, 0.08, 1.32, 0.01, 3.69),                     # T_er
  st0 = rtt(n, 0.17, 0.04, 0.88, 0.00, 4.75))                     # s_T_er

# the only constraint: the starting-point range must stay inside the barriers
feasible <- function(d) d[d$w - d$sw/2 > 0 & d$w + d$sw/2 < 1, ]

g <- feasible(draw(4 * NSET))
stopifnot(nrow(g) >= NSET)
g <- head(g, NSET)
write.csv(g, "bench_sets.csv", row.names = FALSE)

cat(sprintf("%d sets -> bench_sets.csv\n", nrow(g)))
print(round(sapply(g, quantile, c(0, .5, 1)), 3))
