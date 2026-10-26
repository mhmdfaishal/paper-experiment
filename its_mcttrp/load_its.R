# Load the ITS_MC-TTRP sources and the compiled C++ library.
# Usage: source("load_its.R") from the repository root.
options(its_root = normalizePath("."))
library(Rcpp)

# [fix] The code was written for R 4.0.2, where `&&`/`||` silently used the first
# element of a longer vector (e.g. input$capacidad.vehiculo has one identical
# entry per trailer). R >= 4.3 raises an error instead. These wrappers restore
# the R 4.0 behaviour; `y` is a lazy promise, so short-circuiting is preserved.
`&&` <- function(x, y) {
  x <- if (length(x)) as.logical(x[[1]]) else NA
  if (identical(x, FALSE)) return(FALSE)
  y <- if (length(y)) as.logical(y[[1]]) else NA
  if (identical(y, FALSE)) return(FALSE)
  if (is.na(x) | is.na(y)) NA else TRUE
}
`||` <- function(x, y) {
  x <- if (length(x)) as.logical(x[[1]]) else NA
  if (identical(x, TRUE)) return(TRUE)
  y <- if (length(y)) as.logical(y[[1]]) else NA
  if (identical(y, TRUE)) return(TRUE)
  if (is.na(x) | is.na(y)) NA else FALSE
}
for (f in list.files("R", pattern = "\\.R$", full.names = TRUE)) {
  if (basename(f) != "RcppExports.R") source(f)  # RcppExports.R points to an old package name
}
dyn.load(file.path("src", paste0("itsmcttrp", .Platform$dynlib.ext)))
