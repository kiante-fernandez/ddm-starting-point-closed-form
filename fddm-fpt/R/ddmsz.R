# R wrapper for core/ddm_sz.h.  Build, then self-check against WienR's values:
#   PKG_CPPFLAGS=-I../core R CMD SHLIB -o ddmsz.so ../core/ddm_sz.cpp ../core/Faddeeva.cc ddmsz_r.cpp
#   Rscript ddmsz.R
.here <- tryCatch(dirname(normalizePath(sys.frame(1)$ofile)), error = function(e) ".")
dyn.load(file.path(.here, paste0("ddmsz", .Platform$dynlib.ext)))

# Lower barrier, drift ~ N(v, sv^2), relative start ~ U(w - sz/2, w + sz/2) as in WienR's (w, sw).
# gradient rows: v, sv, a, w, sz.  For absolute z and sz, divide both by a first.
ddm_sz <- function(t, v, sv, a, w, sz = 0, tol = 1e-12) {
  r <- .Call("R_ddm_sz", as.double(t), as.double(c(v, sv, a, w, sz, tol)))
  list(density = r[[1]], gradient = r[[2]])
}

# Adds t0 ~ U(t0, t0 + st0); gradient rows v, sv, a, w, sz, t0, st0.
ddm_sz7 <- function(t, v, sv, a, w, sz, t0, st0, tol = 1e-12) {
  r <- .Call("R_ddm_sz7", as.double(t), as.double(c(v, sv, a, w, sz, t0, st0, tol)))
  list(density = r[[1]], gradient = r[[2]])
}

# rtdists parametrization: a full separation, z and sz absolute, t0 the lower edge of the
# non-decision window, response "upper"/"lower".  Gradient rows: a, v, t0, z, sz, sv, st0.
ddm_rtdists <- function(rt, response, a, v, t0, z = a/2, sz = 0, sv = 0, st0 = 0, tol = 1e-12) {
  s <- if (identical(as.character(response)[1], "upper")) -1 else 1
  w <- if (s < 0) 1 - z/a else z/a
  r <- ddm_sz7(rt, s*v, sv, a, w, sz/a, t0, st0, tol)
  d <- r$gradient
  r$gradient <- rbind(a = d[3, ] - s*z/a^2 * d[4, ] - sz/a^2 * d[5, ],
                      v = s * d[1, ], t0 = d[6, ], z = s/a * d[4, ],
                      sz = d[5, ]/a, sv = d[2, ], st0 = d[7, ])
  r
}

if (sys.nframe() == 0L) {                        # the binding: core/check covers the grid itself
  g <- head(read.csv("../../data/wienr_grad.csv"), 20)
  o <- sapply(seq_len(nrow(g)), function(i) with(g[i, ], unlist(ddm_sz(t, v, sv, a, w, sw))))
  ref <- t(as.matrix(g[, c("wienr", "dv", "dsv", "da", "dw", "dsw")]))
  err <- max(abs(o - ref) / pmax(abs(ref), rep(o[1, ], each = 6)))
  cat(sprintf("vs WienR, %d rows through the R binding: max rel %.1e\n", nrow(g), err))
  stopifnot(err < 1e-8)
  cat("R CHECK PASS\n")

  if (requireNamespace("rtdists", quietly = TRUE)) {
  p <- list(a = 1.1, v = 1.3, t0 = 0.25, z = 0.6, sz = 0.2, sv = 0.9, st0 = 0.15)
  rt <- c(0.32, 0.5, 0.9, 1.8)
  for (resp in c("lower", "upper")) {
    o <- do.call(ddm_rtdists, c(list(rt, resp), p))
    ref <- do.call(rtdists::ddiffusion, c(list(rt, resp, precision = 8), p))
    fd <- sapply(names(p), function(k) {
      h <- 1e-6; up <- p; dn <- p; up[[k]] <- p[[k]] + h; dn[[k]] <- p[[k]] - h
      (do.call(ddm_rtdists, c(list(rt, resp), up))$density -
       do.call(ddm_rtdists, c(list(rt, resp), dn))$density) / (2*h)
    })
    e <- max(abs(o$density/ref - 1))
    eg <- max(abs(t(o$gradient[names(p), ]) - fd) / pmax(abs(fd), o$density))
    cat(sprintf("rtdists %s: density %.1e  gradient vs FD %.1e\n", resp, e, eg))
    stopifnot(e < 1e-6, eg < 1e-6)
  }
  cat("RTDISTS CHECK PASS\n")
  }
}
