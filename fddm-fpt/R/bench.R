# Cost and accuracy vs rtdists and WienR, rtdists parametrization, over parameter sets drawn
# (EMC2's dDDM calls the same Hartmann-Klauer engine as WienR, at a default precision of 5e-3)
# from the priors of Tran et al. (2021), see bench_sets.R.  Response times are simulated from
# each set, so both barriers appear in the proportions the model implies.
#   OMP_NUM_THREADS=1 Rscript bench.R  ->  bench_r.csv
suppressMessages({library(rtdists); library(WienR); library(microbenchmark)})
sink(tempfile()); source("ddmsz.R"); sink()

NRT   <- as.integer(Sys.getenv("NRT", 1000))
LIMIT <- as.numeric(Sys.getenv("LIMIT", 30))    # seconds; a slower call is recorded, not waited on
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
capped <- function(expr) {
  setTimeLimit(elapsed = LIMIT, transient = TRUE)
  on.exit(setTimeLimit())
  tryCatch(expr, error = function(e) NULL)
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

# name, function, trials per call, repetitions (the slow settings get fewer of both)
M <- list(
  list("ours 1e-12 (density + 7 grads)", ours,          NRT, 10),
  list("rtdists precision=3",            rtd(3),        NRT, 10),
  list("rtdists precision=5",            rtd(5),        100,  3),
  list("rtdists precision=8",            rtd(8),          5,  2),
  list("WienR 5e-3 (EMC2 default)",      wienr(5e-3),   NRT, 10),
  list("WienR 1e-4",                     wienr(1e-4),   NRT, 10),
  list("WienR 1e-8",                     wienr(1e-8),   NRT, 10),
  list("WienR 1e-12",                    wienr(1e-12),  200,  5),
  list("WienR 1e-12 + 5 grads",          wienrg(1e-12), 200,  3))

head_n <- function(rts, n) lapply(rts, function(x) head(x, max(1, round(n*length(x)/NRT))))

out <- NULL
for (i in seq_along(sets)) {
  p <- sets[[i]]
  rts <- draw(NRT, p)
  ref <- ours(rts, p)
  big <- ref > 1e-10
  for (m in M) {
    nm <- m[[1]]; f <- m[[2]]; n <- min(m[[3]], NRT); reps <- m[[4]]
    r <- head_n(rts, n)
    nn <- sum(lengths(r))
    tm <- capped(microbenchmark(f(r, p), times = reps, unit = "us"))
    us <- if (is.null(tm)) NA else median(tm$time)/1e3/nn
    acc <- if (identical(nm, M[[1]][[1]])) 0 else {
      v <- capped(f(rts, p))
      if (is.null(v)) NA else max(abs(v[big]/ref[big] - 1))
    }
    out <- rbind(out, data.frame(set = i, method = nm, us_per_trial = us, n = nn, rel_diff = acc))
    cat(sprintf("set %2d  %-30s %10s us/trial  rel %.1e\n", i, nm,
                if (is.na(us)) sprintf("> %ds", LIMIT) else sprintf("%.2f", us), acc))
    flush.console()
  }
}
write.csv(out, "bench_r.csv", row.names = FALSE)

cat("\n== median over", NSET, "sets (IQR), accuracy vs ours at 1e-12 ==\n")
for (m in M) {
  s <- out[out$method == m[[1]], ]
  u <- s$us_per_trial[!is.na(s$us_per_trial)]
  cat(sprintf("%-30s %9.2f us  [%7.2f, %8.2f]  timed out %d/%d   rel: median %.1e  max %.1e\n",
              m[[1]], median(u), quantile(u, .25), quantile(u, .75),
              sum(is.na(s$us_per_trial)), nrow(s),
              median(s$rel_diff, na.rm = TRUE), max(s$rel_diff, na.rm = TRUE)))
}
cat(sprintf("\n%s | R %s | rtdists %s | WienR %s\n", Sys.info()[["sysname"]],
            getRversion(), packageVersion("rtdists"), packageVersion("WienR")))
