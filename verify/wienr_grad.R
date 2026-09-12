suppressMessages(library(WienR))
g <- read.csv("wienr_grid.csv")
for (nm in c("v", "sv", "a", "w", "sw")) {
  f <- get(paste0("d", nm, "WienerPDF"))
  g[[paste0("d", nm)]] <- f(g$t, "lower", g$a, g$v, g$w, t0 = 0, sv = g$sv, sw = g$sw, st0 = 0, precision = 1e-12, n.evals = 0)$deriv
}
write.csv(g, "wienr_grad.csv", row.names = FALSE)
