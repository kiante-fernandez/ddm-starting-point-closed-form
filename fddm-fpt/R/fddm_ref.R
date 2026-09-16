# Fixed-start reference from fddm (values only; fddm is GPL-2, no code copied).
#   Rscript fddm_ref.R  ->  fddm_fixed.csv
library(fddm)
g <- read.csv("../../data/wienr_grad.csv")
e <- 1e-12
arg <- list(rt = g$t, response = "lower", v = g$v, a = g$a, t0 = 0, w = g$w, sv = g$sv, err_tol = e)
write.csv(data.frame(t = g$t, a = g$a, v = g$v, w = g$w, sv = g$sv,
                     f   = do.call(dfddm,     arg),
                     dv  = do.call(dv_dfddm,  arg[-8]),
                     dsv = do.call(dsv_dfddm, arg[-8]),
                     da  = do.call(da_dfddm,  arg[-8]),
                     dw  = do.call(dw_dfddm,  arg[-8])),
          "fddm_fixed.csv", row.names = FALSE)
