source("load_its.R")  # [fix] scripts did not load the solver

real_problem <- list()
real_problem[[1]] <- MCTTRP_opt_solver("instances/P1.csv", "MCTTRP", 2, 1, total_time=60, total_iterations=100, verbose=0)
real_problem[[2]] <- MCTTRP_opt_solver("instances/P2.csv", "MCTTRP", 2, 1, total_time=60, total_iterations=100, verbose=0)
real_problem[[3]] <- MCTTRP_opt_solver("instances/P3.csv", "MCTTRP", 2, 1, total_time=60, total_iterations=100, verbose=0)
real_problem[[4]] <- MCTTRP_opt_solver("instances/P4.csv", "MCTTRP", 2, 1, total_time=60, total_iterations=100, verbose=0)
real_problem[[5]] <- MCTTRP_opt_solver("instances/P5.csv", "MCTTRP", 2, 1, total_time=60, total_iterations=100, verbose=0)
real_problem[[6]] <- MCTTRP_opt_solver("instances/P6.csv", "MCTTRP", 2, 1, total_time=60, total_iterations=100, verbose=0)
real_problem[[7]] <- MCTTRP_opt_solver("instances/P7.csv", "MCTTRP", 2, 1, total_time=60, total_iterations=100, verbose=0)
real_problem[[8]] <- MCTTRP_opt_solver("instances/P8.csv", "MCTTRP", 2, 1, total_time=60, total_iterations=100, verbose=0)

for (i in 1:length(real_problem)) {  # [fix] was length(result), undefined here
  print(paste0(i,"  cost: ", real_problem[[i]]$cost))
}