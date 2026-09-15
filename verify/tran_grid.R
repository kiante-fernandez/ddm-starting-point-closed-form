# Literature grid: 2000 parameter sets drawn from the informative priors of Tran, van Maanen,
# Heathcote & Matzke (2021, Front. Psychol., Table 1; s = 1 units), truncated at their empirical
# bounds, plus fixed points at the bounds themselves.  Run from data/:  Rscript ../verify/tran_grid.R
suppressMessages(library(WienR))
set.seed(7); n <- 2000
rtn <- function(n, m, s, lo, hi) { x <- rnorm(4*n, m, s); x <- x[x > lo & x < hi]; x[seq_len(n)] }
rtt <- function(n, m, s, df, lo, hi) { x <- m + s*rt(20*n, df); x <- x[x > lo & x < hi]; x[seq_len(n)] }
draw <- function(n) data.frame(
  t  = runif(n, 0.05, 3.0),
  a  = pmin(pmax(rgamma(n, 11.69, scale = 0.12), 0.11), 7.47),
  v  = sample(c(-1, 1), n, TRUE) * rtn(n, 1.76, 1.51, 0.01, 18.51),
  w  = rtt(n, 0.5, 0.05, 1.85, 0.04, 0.96),
  sv = rtn(n, 1.36, 0.69, 0.0, 3.45),
  sw = rtn(n, 0.33, 0.22, 0.01, 0.85))
feasible <- function(d) d[d$w - d$sw/2 > 0 & d$w + d$sw/2 < 1, ]   # the range must stay inside the barriers
g <- feasible(draw(2 * n))[seq_len(n), ]                            # resample so exactly n sets survive
# corners: each parameter at an empirical bound, the others typical; three t each
typ <- list(a = 1.4, v = 1.76, w = 0.5, sv = 1.36, sw = 0.33)
corner <- function(...) { p <- modifyList(typ, list(...)); data.frame(t = c(0.2, 1, 3), p) }
g <- rbind(g, corner(a = 7.47), corner(a = 0.11), corner(v = 18.51), corner(v = -18.51), corner(sv = 3.45),
           corner(sw = 0.85), corner(sw = 0.01), corner(w = 0.04, sw = 0.06), corner(w = 0.96, sw = 0.06),
           corner(a = 7.47, v = -18.51, sv = 3.45, sw = 0.85))
g <- feasible(g)
g$wienr <- WienerPDF(g$t, "lower", g$a, g$v, g$w, t0 = 0, sv = g$sv, sw = g$sw, st0 = 0, precision = 1e-12, n.evals = 0)$value
for (nm in c("v", "sv", "a", "w", "sw")) {
  f <- get(paste0("d", nm, "WienerPDF"))
  g[[paste0("d", nm)]] <- f(g$t, "lower", g$a, g$v, g$w, t0 = 0, sv = g$sv, sw = g$sw, st0 = 0, precision = 1e-12, n.evals = 0)$deriv
}
write.csv(g, "tran_grid.csv", row.names = FALSE)
cat(nrow(g), "sets written to tran_grid.csv\n")
