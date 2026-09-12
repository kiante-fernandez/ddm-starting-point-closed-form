suppressMessages(library(WienR))
set.seed(11); n <- 200
g <- data.frame(t = runif(n, 0.35, 3.0), a = sample(c(0.8,1.2,2.0), n, TRUE),
  v = sample(c(-2,-0.5,0,1,3), n, TRUE), w = sample(c(0.35,0.5,0.65), n, TRUE),
  sv = sample(c(0.5,1.5), n, TRUE), sw = sample(c(0.1,0.3), n, TRUE), st0 = sample(c(0.1,0.25), n, TRUE))
t0 <- Sys.time()
r <- WienerPDF(g$t, "lower", g$a, g$v, g$w, t0 = 0.3, sv = g$sv, sw = g$sw, st0 = g$st0, precision = 1e-10, n.evals = 0)
el <- as.numeric(Sys.time()-t0, units="secs")
g$wienr <- r$value; g$err <- r$err
write.csv(g, "wienr_full.csv", row.names = FALSE)
r9 <- WienerPDF(g$t, "lower", g$a, g$v, g$w, t0 = 0.3, sv = g$sv, sw = g$sw, st0 = g$st0, precision = 1e-9, n.evals = 0)
write.csv(data.frame(w9 = r9$value, err9 = r9$err), "wienr_full_p9.csv", row.names = FALSE)   # same sets at tolerance 1e-9, for the matched-accuracy row of the timing table
cat(sprintf("WienR 7-param: %.0f us per evaluation, max reported err %.1e\n", 1e6*el/n, max(r$err)))
