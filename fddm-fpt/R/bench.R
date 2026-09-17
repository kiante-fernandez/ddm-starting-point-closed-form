# Cost and accuracy vs rtdists and WienR, rtdists parametrization, over parameter sets drawn
# (EMC2's dDDM calls the same Hartmann-Klauer engine as WienR, at a default precision of 5e-3)
# from the priors of Tran et al. (2021), see bench_sets.R.  Response times are simulated from
# each set, so both barriers appear in the proportions the model implies.
#   OMP_NUM_THREADS=1 Rscript bench.R          ->  bench_r.csv
#   SWEEP=1 OMP_NUM_THREADS=1 Rscript bench.R  ->  curve_r.csv  (cost vs trials per evaluation)
suppressMessages({library(rtdists); library(WienR); library(microbenchmark)})
sink(tempfile()); source("ddmsz.R"); sink()

NRT <- as.integer(Sys.getenv("NRT", 1000))
set.seed(1)

g <- read.csv("bench_sets.csv")                 # Tran et al. priors, all seven parameters
NSET <- nrow(g)
sets <- lapply(seq_len(NSET), function(i) list(
  a = g$a[i], v = g$v[i], z = g$w[i] * g$a[i], sz = g$sw[i] * g$a[i], sv = g$sv[i],
  t0 = g$t0[i], st0 = g$st0[i]))

draw <- function(n, p) {                        # rts from the model, split by barrier
  d <- do.call(rdiffusion, c(list(n = n), p))
  split(d$rt, factor(d$response, c("lower", "upper")))
}
wpar <- function(p) list(a = p$a, v = p$v, w = p$z/p$a, t0 = p$t0, sv = p$sv,
                         sw = p$sz/p$a, st0 = p$st0)

# each method evaluates the whole data set: one call per barrier
byresp <- function(one) function(rts, p) unlist(lapply(names(rts), function(r)
  if (length(rts[[r]])) one(rts[[r]], r, p) else numeric(0)))

ours   <- byresp(function(rt, r, p) do.call(ddm_rtdists, c(list(rt, r), p))$density)
rtd    <- function(pr) byresp(function(rt, r, p)
  do.call(ddiffusion, c(list(rt, r, precision = pr), p)))
wienr  <- function(pr) byresp(function(rt, r, p)
  do.call(WienerPDF, c(list(rt, r), wpar(p), list(precision = pr, n.threads = FALSE)))$value)
wienrg <- function(pr) byresp(function(rt, r, p) {
  v <- do.call(WienerPDF, c(list(rt, r), wpar(p), list(precision = pr, n.threads = FALSE)))$value
  for (nm in c("a", "v", "w", "sv", "sw"))
    do.call(get(paste0("d", nm, "WienerPDF")),
            c(list(rt, r), wpar(p), list(precision = pr, n.threads = FALSE)))
  v
})

# name, function, trials per timing call, repetitions, response times used for accuracy
M <- list(
  list("ours 1e-12 (density + 7 grads)", ours,          NRT, 10, NRT),
  list("rtdists precision=3",            rtd(3),        NRT, 10, NRT),
  list("rtdists precision=5",            rtd(5),        100,  3, 100),
  list("rtdists precision=8",            rtd(8),         25,  2,  50),
  list("WienR 5e-3 (EMC2 default)",      wienr(5e-3),   NRT, 10, NRT),
  list("WienR 1e-4",                     wienr(1e-4),   NRT, 10, NRT),
  list("WienR 1e-8",                     wienr(1e-8),   NRT, 10, NRT),
  list("WienR 1e-12",                    wienr(1e-12),  200,  5, 200),
  list("WienR 1e-12 + 5 grads",          wienrg(1e-12), 200,  3, 200))

take <- function(rts, n) lapply(rts, function(x) head(x, max(1, round(n*length(x)/NRT))))

if (nzchar(Sys.getenv("SWEEP"))) {       # median over the first 5 sets, stop once a call passes 10 s
  sweep <- NULL
  for (m in M[-c(4, 6)]) {               # rtdists precision 8 and WienR 1e-4 add no new line shape
    for (n in c(30, 100, 300, 1000, 3000, 10000, 30000, 100000)) {
      ms <- median(sapply(sets[1:5], function(p) {
        rts <- draw(n, p)                 # simulate outside: microbenchmark re-evaluates its argument
        median(microbenchmark(m[[2]](rts, p), times = 3)$time) / 1e6
      }))
      sweep <- rbind(sweep, data.frame(method = m[[1]], n = n, ms = ms))
      cat(sprintf("%-30s n=%7d  %10.3f ms\n", m[[1]], n, ms)); flush.console()
      if (ms > 1e4) break
    }
  }
  write.csv(sweep, "curve_r.csv", row.names = FALSE)
  quit(save = "no")
}

out <- NULL
t_start <- Sys.time()
unlink("bench_r.csv")                              # rows are appended below, set by set
for (i in seq_along(sets)) {
  p <- sets[[i]]
  rts <- draw(NRT, p)
  for (m in M) {
    nm <- m[[1]]; f <- m[[2]]; reps <- m[[4]]
    r <- take(rts, min(m[[3]], NRT))
    nn <- sum(lengths(r))
    us <- median(microbenchmark(f(r, p), times = reps, unit = "us")$time)/1e3/nn
    k <- take(rts, min(m[[5]], NRT))
    acc <- if (identical(nm, M[[1]][[1]])) 0 else {
      v <- f(k, p); rf <- ours(k, p)          # compare on the same response times
      max(abs(v[rf > 1e-10]/rf[rf > 1e-10] - 1))
    }
    row <- data.frame(set = i, method = nm, us_per_trial = us, n = nn,
                      n_acc = sum(lengths(k)), rel_diff = acc)
    out <- rbind(out, row)
    write.table(row, "bench_r.csv", sep = ",", row.names = FALSE, append = TRUE,
                col.names = (i == 1 && identical(nm, M[[1]][[1]])))
    cat(sprintf("[%4.0f min] set %3d/%d  %-30s %10.2f us/trial  rel %.1e\n",
                as.numeric(difftime(Sys.time(), t_start, units = "mins")), i, length(sets), nm, us, acc))
    flush.console()
  }
}

cat("\n== median over", NSET, "sets (IQR), accuracy vs ours at 1e-12 ==\n")
for (m in M) {
  s <- out[out$method == m[[1]], ]
  u <- s$us_per_trial
  cat(sprintf("%-30s %10.2f us  [%8.2f, %9.2f]  %d sets   rel: median %.1e  max %.1e\n",
              m[[1]], median(u), quantile(u, .25), quantile(u, .75), nrow(s),
              median(s$rel_diff), max(s$rel_diff)))
}
cat(sprintf("\n%s | R %s | rtdists %s | WienR %s\n", Sys.info()[["sysname"]],
            getRversion(), packageVersion("rtdists"), packageVersion("WienR")))
