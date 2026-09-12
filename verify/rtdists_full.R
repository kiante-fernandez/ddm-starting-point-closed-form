suppressMessages(library(rtdists))
g <- read.csv("wienr_full.csv")   # 200 sets, t0=0.3, st0 in {0.1,0.25}
# WienR: t0 ~ U(t0, t0+st0).  rtdists: t0 ~ U(t0, t0+st0) as well (mean t0+st0/2), so t0 matches directly.
for (p in c(3,5,6)) {
  t0 <- Sys.time()
  v <- ddiffusion(g$t, "lower", a=g$a, v=g$v, t0=0.3, z=g$w*g$a, sz=g$sw*g$a, sv=g$sv, st0=g$st0, s=1, precision=p)
  cat(sprintf("rtdists 7-param precision=%d: %.0f us/eval\n", p, 1e6*as.numeric(Sys.time()-t0,units="secs")/nrow(g)))
  g[[paste0("rtd",p)]] <- v
}
write.csv(g, "rtdists_full.csv", row.names=FALSE)
