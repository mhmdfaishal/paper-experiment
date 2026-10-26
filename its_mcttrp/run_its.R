# Run one ITS_MC-TTRP experiment and save the result as JSON.
# Usage (from the repository root):
#   Rscript run_its.R <instance> <n_trucks> <n_trailers> <iterations> <max_time_s> <seed> <out.json>
args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 7)
inst <- args[1]; n_trucks <- as.integer(args[2]); n_trailers <- as.integer(args[3])
iters <- as.integer(args[4]); max_time <- as.numeric(args[5]); seed <- as.integer(args[6])
out <- args[7]

suppressMessages(source("load_its.R"))

# the solver prints a lot of progress lines; keep them out of the console
log_file <- sub("\\.json$", ".log", out)
sink(log_file)
res <- MCTTRP_opt_solver(inst, "MCTTRP", n_trucks, n_trailers,
                         total_time = max_time, total_iterations = iters,
                         verbose = 0, seed = seed)
sink()

# minimal JSON writer (avoids an extra package dependency)
num <- function(x) paste0("[", paste(format(x, digits = 15, scientific = FALSE, trim = TRUE), collapse = ","), "]")
routes <- sapply(res$result_res, function(r)
  sprintf('{"type":"%s","route":%s}', r$type, num(r$route)))
tr <- get0(".its_trace", envir = .GlobalEnv, ifnotfound = data.frame(time = 0, iter = 0, best = res$cost))
json <- sprintf(paste0('{"instance":"%s","seed":%d,"iterations":%d,"cost":%s,"init_cost":%s,',
                       '"time":%s,"routes":[%s],"trace":{"time":%s,"iter":%s,"best":%s}}'),
                basename(inst), seed, iters,
                format(res$cost, digits = 15), format(res$init_cost, digits = 15),
                format(as.numeric(res$time), digits = 10),
                paste(routes, collapse = ","), num(tr$time), num(tr$iter), num(tr$best))
writeLines(json, out)
cat(sprintf("%s seed=%d cost=%.4f time=%.1fs\n", basename(inst), seed, res$cost, as.numeric(res$time)))
