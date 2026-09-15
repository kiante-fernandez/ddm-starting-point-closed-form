suppressMessages(library(rtdists))
g <- read.csv("wienr_grad.csv")
g$p3 <- ddiffusion(g$t, "lower", a=g$a, v=g$v, t0=0, z=g$w*g$a, sz=g$sw*g$a, sv=g$sv, st0=0, s=1, precision=3)
write.csv(g, "rtdists_prec.csv", row.names=FALSE)
