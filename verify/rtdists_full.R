suppressMessages(library(rtdists))
g <- read.csv("wienr_full.csv")   # the 400 sets with t0 = 0.3, st0 in {0.1, 0.25}
# WienR and rtdists both take t0 ~ U(t0, t0 + st0), so t0 matches directly.
g$rtd3 <- ddiffusion(g$t, "lower", a=g$a, v=g$v, t0=0.3, z=g$w*g$a, sz=g$sw*g$a, sv=g$sv, st0=g$st0, s=1, precision=3)
write.csv(g, "rtdists_full.csv", row.names=FALSE)
