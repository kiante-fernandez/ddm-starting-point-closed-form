suppressMessages(library(WienR))
g <- read.csv("wienr_grad.csv")[, c("t", "a", "v", "w", "sv", "sw")]   # the six-parameter sets, plus t0 and st0
set.seed(11); g$st0 <- sample(c(0.1, 0.25), nrow(g), TRUE); g$t <- g$t + 0.3   # t0 = 0.3
t0 <- Sys.time()
r <- WienerPDF(g$t, "lower", g$a, g$v, g$w, t0 = 0.3, sv = g$sv, sw = g$sw, st0 = g$st0, precision = 1e-12, n.evals = 0)
el <- as.numeric(Sys.time()-t0, units="secs")
g$wienr <- r$value
write.csv(g, "wienr_full.csv", row.names = FALSE)
cat(sprintf("WienR 7-param: %.0f us per evaluation, max reported err %.1e\n", 1e6*el/nrow(g), max(r$err)))
