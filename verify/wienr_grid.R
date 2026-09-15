suppressMessages(library(WienR))
set.seed(3)
n <- 400
grid <- data.frame(
  t  = runif(n, 0.05, 3.0),
  a  = sample(c(0.6, 1.0, 1.5, 2.5), n, TRUE),
  v  = sample(c(-3, -1, -0.2, 0, 0.5, 1.5, 4), n, TRUE),
  w  = sample(c(0.3, 0.5, 0.7), n, TRUE),
  sv = sample(c(0.5, 1.0, 2.0), n, TRUE),
  sw = sample(c(0.05, 0.2, 0.4), n, TRUE))
t0 <- Sys.time()
res <- WienerPDF(grid$t, "lower", grid$a, grid$v, grid$w, t0 = 0,
                 sv = grid$sv, sw = grid$sw, st0 = 0, precision = 1e-12, n.evals = 0)
el <- as.numeric(Sys.time() - t0, units = "secs")
grid$wienr <- res$value
for (nm in c("v", "sv", "a", "w", "sw")) {
  f <- get(paste0("d", nm, "WienerPDF"))
  grid[[paste0("d", nm)]] <- f(grid$t, "lower", grid$a, grid$v, grid$w, t0 = 0,
                               sv = grid$sv, sw = grid$sw, st0 = 0, precision = 1e-12, n.evals = 0)$deriv
}
write.csv(grid, "wienr_grad.csv", row.names = FALSE)
cat(sprintf("WienR: %d evaluations in %.2f s (%.0f us each); max reported integration err %.1e\n",
            n, el, 1e6*el/n, max(res$err)))
