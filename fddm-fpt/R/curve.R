# Wall time per likelihood evaluation vs trials per evaluation, R side.  Median over 5 published
# parameter sets, response times simulated from each.
#   OMP_NUM_THREADS=1 Rscript curve.R  ->  curve_r.csv
suppressMessages({library(rtdists); library(WienR); library(microbenchmark)})
sink(tempfile()); source("ddmsz.R"); sink()

N <- c(30, 100, 300, 1000, 3000, 10000, 30000, 100000)
BUDGET <- 10; NSET <- 5
set.seed(1)

g <- head(read.csv("bench_sets.csv"), NSET)
sets <- lapply(seq_len(NSET), function(i) list(
  a = g$a[i], v = g$v[i], z = g$w[i] * g$a[i], sz = g$sw[i] * g$a[i], sv = g$sv[i],
  t0 = g$t0[i], st0 = g$st0[i]))

draw <- function(n, p) {
  d <- do.call(rdiffusion, c(list(n = n), p))
  split(d$rt, factor(d$response, c("lower", "upper")))
}
wpar <- function(p) list(a = p$a, v = p$v, w = p$z/p$a, t0 = p$t0, sv = p$sv,
                         sw = p$sz/p$a, st0 = p$st0)
byresp <- function(one) function(rts, p) for (r in names(rts))
  if (length(rts[[r]])) one(rts[[r]], r, p)

methods <- list(
  `ours, density + 7 gradients (R)` = byresp(function(rt, r, p)
    do.call(ddm_rtdists, c(list(rt, r), p))),
  `rtdists, precision 3` = byresp(function(rt, r, p)
    do.call(ddiffusion, c(list(rt, r, precision = 3), p))),
  `rtdists, precision 5` = byresp(function(rt, r, p)
    do.call(ddiffusion, c(list(rt, r, precision = 5), p))),
  `WienR, 5e-3 (EMC2 default)` = byresp(function(rt, r, p)
    do.call(WienerPDF, c(list(rt, r), wpar(p), list(precision = 5e-3, n.threads = FALSE)))),
  `WienR, 1e-8` = byresp(function(rt, r, p)
    do.call(WienerPDF, c(list(rt, r), wpar(p), list(precision = 1e-8, n.threads = FALSE)))),
  `WienR, 1e-12` = byresp(function(rt, r, p)
    do.call(WienerPDF, c(list(rt, r), wpar(p), list(precision = 1e-12, n.threads = FALSE)))),
  `WienR, 1e-12 + 5 gradients` = byresp(function(rt, r, p) {
    do.call(WienerPDF, c(list(rt, r), wpar(p), list(precision = 1e-12, n.threads = FALSE)))
    for (nm in c("a", "v", "w", "sv", "sw"))
      do.call(get(paste0("d", nm, "WienerPDF")),
              c(list(rt, r), wpar(p), list(precision = pr <- 1e-12, n.threads = FALSE)))
  }))

out <- NULL
for (m in names(methods)) {
  f <- methods[[m]]
  for (n in N) {
    ms <- sapply(sets, function(p) {
      rts <- draw(n, p)
      reps <- if (n <= 1000) 5 else 3
      median(microbenchmark(f(rts, p), times = reps)$time) / 1e6
    })
    med <- median(ms)
    out <- rbind(out, data.frame(method = m, n = n, ms = med))
    cat(sprintf("%-32s n=%7d  %10.3f ms\n", m, n, med)); flush.console()
    if (med/1e3 > BUDGET) break
  }
}
write.csv(out, "curve_r.csv", row.names = FALSE)
cat("wrote curve_r.csv\n")
