source("load_its.R")  # [fix] scripts did not load the solver


result_ITS <- list()
result_ITS[[1]] <- MCTTRP_opt_solver("instances/CHAO_MCTTRP_01.txt", "MCTTRP", 8, 5, total_time=60000, total_iterations=100, verbose=0)
result_ITS[[2]] <- MCTTRP_opt_solver("instances/CHAO_MCTTRP_02.txt", "MCTTRP", 8, 5, total_time=60000, total_iterations=100, verbose=0)
result_ITS[[3]] <- MCTTRP_opt_solver("instances/CHAO_MCTTRP_03.txt", "MCTTRP", 8, 5, total_time=60000, total_iterations=100, verbose=0) 

result_ITS[[4]] <- MCTTRP_opt_solver("instances/CHAO_MCTTRP_04.txt", "MCTTRP", 13, 8, total_time=60000, total_iterations=100, verbose=0)
result_ITS[[5]] <- MCTTRP_opt_solver("instances/CHAO_MCTTRP_05.txt", "MCTTRP", 14, 8, total_time=60000, total_iterations=100, verbose=0)
result_ITS[[6]] <- MCTTRP_opt_solver("instances/CHAO_MCTTRP_06.txt", "MCTTRP", 14, 8, total_time=60000, total_iterations=100, verbose=0) 

result_ITS[[7]] <- MCTTRP_opt_solver("instances/CHAO_MCTTRP_07.txt", "MCTTRP", 11, 6, total_time=60000, total_iterations=100, verbose=0)
result_ITS[[8]] <- MCTTRP_opt_solver("instances/CHAO_MCTTRP_08.txt", "MCTTRP", 12, 6, total_time=60000, total_iterations=100, verbose=0)
result_ITS[[9]] <- MCTTRP_opt_solver("instances/CHAO_MCTTRP_09.txt", "MCTTRP", 11, 6, total_time=60000, total_iterations=100, verbose=0)

result_ITS[[10]] <- MCTTRP_opt_solver("instances/CHAO_MCTTRP_10.txt", "MCTTRP", 18, 9, total_time=60000, total_iterations=100, verbose=0)
result_ITS[[11]] <- MCTTRP_opt_solver("instances/CHAO_MCTTRP_11.txt", "MCTTRP", 18, 9, total_time=60000, total_iterations=100, verbose=0)
result_ITS[[12]] <- MCTTRP_opt_solver("instances/CHAO_MCTTRP_12.txt", "MCTTRP", 18, 9, total_time=60000, total_iterations=100, verbose=0)

result_ITS[[13]] <- MCTTRP_opt_solver("instances/CHAO_MCTTRP_13.txt", "MCTTRP", 23, 14, total_time=60000, total_iterations=2, verbose=0)
result_ITS[[14]] <- MCTTRP_opt_solver("instances/CHAO_MCTTRP_14.txt", "MCTTRP", 26, 14, total_time=60000, total_iterations=2, verbose=0)
result_ITS[[15]] <- MCTTRP_opt_solver("instances/CHAO_MCTTRP_15.txt", "MCTTRP", 25, 13, total_time=60000, total_iterations=2, verbose=0)

result_ITS[[16]] <- MCTTRP_opt_solver("instances/CHAO_MCTTRP_16.txt", "MCTTRP", 11, 6, total_time=60000, total_iterations=2, verbose=0)
result_ITS[[17]] <- MCTTRP_opt_solver("instances/CHAO_MCTTRP_17.txt", "MCTTRP", 11, 6, total_time=60000, total_iterations=2, verbose=0)
result_ITS[[18]] <- MCTTRP_opt_solver("instances/CHAO_MCTTRP_18.txt", "MCTTRP", 11, 6, total_time=60000, total_iterations=2, verbose=0)

result_ITS[[19]] <- MCTTRP_opt_solver("instances/CHAO_MCTTRP_19.txt", "MCTTRP", 13, 8, total_time=60000, total_iterations=2, verbose=0)
result_ITS[[20]] <- MCTTRP_opt_solver("instances/CHAO_MCTTRP_20.txt", "MCTTRP", 13, 8, total_time=60000, total_iterations=2, verbose=0)
result_ITS[[21]] <- MCTTRP_opt_solver("instances/CHAO_MCTTRP_21.txt", "MCTTRP", 13, 8, total_time=60000, total_iterations=2, verbose=0)

for (i in 1:length(result_ITS)) {
  print(paste0(i,"  cost: ", result_ITS[[i]]$cost))
}