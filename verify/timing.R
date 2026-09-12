# WienR cost per evaluation, single thread, 10 000 RTs at one parameter set (the fitting case).
# Same parameter set as verify/timing.py.  Run from anywhere: Rscript verify/timing.R
suppressMessages(library(WienR)); set.seed(1); n <- 10000
t <- runif(n, 0.1, 3); a <- 1.2; v <- 1.0; w <- 0.5; sv <- 1.0; sw <- 0.2; st0 <- 0.2
us <- function(f, reps = 5) { best <- Inf; for (i in 1:reps) { t0 <- Sys.time(); f(); best <- min(best, as.numeric(Sys.time() - t0, units = "secs")) }; 1e6 * best / n }
cat(sprintf("WienR %-42s %8.2f us/eval\n", "sv only (its closed form, Blurton 2017)", us(function() WienerPDF(t, "lower", a, v, w, sv = sv, sw = 0, precision = 1e-12))))
cat(sprintf("WienR %-42s %8.2f us/eval\n", "six-parameter density (sv, sw)", us(function() WienerPDF(t, "lower", a, v, w, sv = sv, sw = sw, precision = 1e-12, n.evals = 0))))
cat(sprintf("WienR %-42s %8.2f us/eval\n", "five gradients (sv, sw)", us(function() for (f in c(dvWienerPDF, dsvWienerPDF, daWienerPDF, dwWienerPDF, dswWienerPDF)) f(t, "lower", a, v, w, sv = sv, sw = sw, precision = 1e-12, n.evals = 0))))
cat(sprintf("WienR %-42s %8.2f us/eval\n", "seven-parameter density (sv, sw, st0)", us(function() WienerPDF(t, "lower", a, v, w, sv = sv, sw = sw, st0 = st0, precision = 1e-12, n.evals = 0), reps = 2)))
suppressMessages(library(rtdists))   # default precision = 3; absolute z = a*w, sz = a*sw
cat(sprintf("rtdists %-40s %8.2f us/eval\n", "six-parameter density (precision 3)", us(function() ddiffusion(t, "lower", a = a, v = v, t0 = 0, z = w * a, sz = sw * a, sv = sv, st0 = 0, s = 1, precision = 3))))
cat(sprintf("rtdists %-40s %8.2f us/eval\n", "seven-parameter density (precision 3)", us(function() ddiffusion(t, "lower", a = a, v = v, t0 = 0, z = w * a, sz = sw * a, sv = sv, st0 = st0, s = 1, precision = 3), reps = 2)))
