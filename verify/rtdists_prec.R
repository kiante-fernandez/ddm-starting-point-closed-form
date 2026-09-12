suppressMessages(library(rtdists))
g <- read.csv("wienr_grid.csv")
out <- data.frame(prec = c(3,4,5,6,8), us = NA, stringsAsFactors = FALSE)
for (i in seq_len(nrow(out))) {
  p <- out$prec[i]; t0 <- Sys.time()
  v <- ddiffusion(g$t, "lower", a=g$a, v=g$v, t0=0, z=g$w*g$a, sz=g$sw*g$a, sv=g$sv, st0=0, s=1, precision=p)
  out$us[i] <- 1e6*as.numeric(Sys.time()-t0, units="secs")/nrow(g)
  g[[paste0("p",p)]] <- v
}
print(out); write.csv(g, "rtdists_prec.csv", row.names=FALSE)
