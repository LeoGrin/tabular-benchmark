source("analyses/random_search_utils.R")
source("analyses/plot_utils.R")


plot_results_per_dataset_bo <- function(df_total, score="accuracy", quantile=0.1, truncate_scores=F, y_inf = 0.5) {
  # Compute the quantile on random + BO
  res_datasets <- 
    df_total %>% 
    filter(!is.na(mean_time))
  
  if (truncate_scores)
    res_datasets <- res_datasets %>% mutate(mean_test_score = if_else(mean_test_score < 0, 0, mean_test_score)) 
  
  # Separate bo and random, as one will be shuffled but not the other
  res_datasets_bo <- res_datasets %>% 
    filter(bo == T)
  res_datasets <- res_datasets %>% 
    filter(bo == F)
  
  
  res_datasets <- res_datasets %>% 
    random_search_no_variable(n_shuffles=15, default_first = T) 
  
  res_datasets_bo <- res_datasets_bo %>% 
    mutate(random_rank = step) %>% 
    group_by(model_name, data__keyword) %>% 
    arrange(random_rank) %>% 
    mutate(best_val_score_so_far = cummax(mean_val_score)) %>% 
    mutate(mean_test_score = if_else(mean_val_score == best_val_score_so_far, mean_test_score, NA_real_)) %>% 
    # Only keep the scores for the first time we beat valid score, put the other to NA
    group_by(model_name, data__keyword, mean_val_score) %>% 
    mutate(mean_test_score = if_else(random_rank == min(random_rank), mean_test_score, NA_real_)) %>% 
    ungroup(mean_val_score) %>% 
    #for every dataset and {{variable}}, prolong the curve by the last best value when there is no value left
    ungroup() %>% 
    complete(model_name, data__keyword, random_rank) %>%
    group_by(model_name, data__keyword) %>% 
    mutate(mean_test_score = na.locf(mean_test_score, na.rm=F))%>% 
    ungroup()
  
  
  
  res_datasets_ <- res_datasets %>% 
    group_by(random_rank, model_name, data__keyword) %>% 
    #summarise(mean_test_score = mean(mean_test_score, na.rm=T)) %>% #summarise on datasets
    summarise(min_test_score = min(mean_test_score, na.rm=T),#quantile(mean_test_score, 0.1, na.rm=T),
              max_test_score = max(mean_test_score, na.rm=T),#quantile(mean_test_score, 0.9, na.rm=T),
              mean_test_score = mean(mean_test_score, na.rm=T))
  
  res_datasets_bo_ <- res_datasets_bo %>% 
    group_by(random_rank, model_name)# %>% 
    #summarise(mean_test_score = mean(mean_test_score, na.rm=T)) #summarise on datasets
  #summarise(min_test_score = min(mean_test_score, na.rm=T),#quantile(mean_test_score, 0.1, na.rm=T),
  #          max_test_score = max(mean_test_score, na.rm=T),#quantile(mean_test_score, 0.9, na.rm=T),
  #          mean_test_score = mean(mean_test_score, na.rm=T))
  
  
  res_datasets_ %>% 
    filter(random_rank <= 200) %>% 
    ggplot() +
    geom_hline(aes(yintercept=mean_test_score, color=model_name), data=res_datasets_ %>% filter(random_rank == 0), linetype="dotted", size=1.5, alpha=0.7) +
    geom_line(aes(x = random_rank, y = mean_test_score, color = model_name), size=2) +
    geom_ribbon(aes(x=random_rank, ymin = min_test_score, ymax=max_test_score, fill=model_name), alpha=0.3) +
    geom_line(aes(x = random_rank, y = mean_test_score, color=model_name), size=2, data=res_datasets_bo_ %>% filter(random_rank <= 200), linetype="11") +
    scale_x_log10() +
    facet_wrap(~data__keyword, scales="free") +
    xlab("Number of random search iterations") +
    ylab(glue("Normalized test {score} of best  \n model (on valid set) up to this iteration")) +
    theme_minimal(base_size=22) +
    theme(legend.position="none") +
    colScale
}


plot_aggregated_results_bo <- function(df_total, score="accuracy", quantile=0.1, truncate_scores=F, y_inf = 0.5) {

  # Compute the quantile on random + BO
  res_datasets <- 
    df_total %>% 
    filter(!is.na(mean_time)) %>% 
    group_by(data__keyword) %>% 
    mutate(old_mean_val_score = mean_val_score, old_mean_test_score = mean_test_score,
           qval = quantile(mean_val_score, quantile, na.rm=T), qtest = quantile(mean_test_score, quantile, na.rm=T)) %>% 
    mutate(mean_val_score = (mean_val_score - quantile(mean_val_score, quantile, na.rm=T)) / (max(mean_val_score, na.rm=T) - quantile(mean_val_score, quantile, na.rm=T)), 
           mean_test_score = (mean_test_score - quantile(mean_test_score, quantile, na.rm=T)) / (max(mean_test_score, na.rm=T) - quantile(mean_test_score, quantile, na.rm=T))) %>% 
    select(model_name, data__keyword, qval, qtest, old_mean_val_score, old_mean_test_score, mean_val_score, mean_test_score, mean_time, step, bo, hp) 
  
  
  if (truncate_scores)
    res_datasets <- res_datasets %>% mutate(mean_test_score = if_else(mean_test_score < 0, 0, mean_test_score)) 
  
  # Separate bo and random, as one will be shuffled but not the other
  res_datasets_bo <- res_datasets %>% 
    filter(bo == T)
  res_datasets <- res_datasets %>% 
    filter(bo == F)
  
  
  res_datasets <- res_datasets %>% 
    random_search_no_variable(n_shuffles=15, default_first = T) 
  
  res_datasets_bo <- res_datasets_bo %>% 
                                mutate(random_rank = step) %>% 
                                group_by(model_name, data__keyword) %>% 
                                arrange(random_rank) %>% 
                                mutate(best_val_score_so_far = cummax(mean_val_score)) %>% 
                                mutate(mean_test_score = if_else(mean_val_score == best_val_score_so_far, mean_test_score, NA_real_)) %>% 
                                # Only keep the scores for the first time we beat valid score, put the other to NA
                                group_by(model_name, data__keyword, mean_val_score) %>% 
                                mutate(mean_test_score = if_else(random_rank == min(random_rank), mean_test_score, NA_real_)) %>% 
                                ungroup(mean_val_score) %>% 
                                #for every dataset and {{variable}}, prolong the curve by the last best value when there is no value left
                                ungroup() %>% 
                                complete(model_name, data__keyword, random_rank) %>%
                                group_by(model_name, data__keyword) %>% 
                                mutate(mean_test_score = na.locf(mean_test_score, na.rm=F))%>% 
                                ungroup()
  
  
  
  res_datasets_ <- res_datasets %>% 
    group_by(random_rank, model_name, n_dataset) %>% 
    summarise(mean_test_score = mean(mean_test_score, na.rm=T)) %>% #summarise on datasets
    summarise(min_test_score = min(mean_test_score, na.rm=T),#quantile(mean_test_score, 0.1, na.rm=T),
              max_test_score = max(mean_test_score, na.rm=T),#quantile(mean_test_score, 0.9, na.rm=T),
              mean_test_score = mean(mean_test_score, na.rm=T))
  
  res_datasets_bo_ <- res_datasets_bo %>% 
    group_by(random_rank, model_name) %>% 
    summarise(mean_test_score = mean(mean_test_score, na.rm=T)) #summarise on datasets
    #summarise(min_test_score = min(mean_test_score, na.rm=T),#quantile(mean_test_score, 0.1, na.rm=T),
    #          max_test_score = max(mean_test_score, na.rm=T),#quantile(mean_test_score, 0.9, na.rm=T),
    #          mean_test_score = mean(mean_test_score, na.rm=T))
  
  
  res_datasets_ %>% 
    filter(random_rank <= 200) %>% 
    ggplot() +
    geom_hline(aes(yintercept=mean_test_score, color=model_name), data=res_datasets_ %>% filter(random_rank == 0), linetype="dotted", size=1.5, alpha=0.7) +
    geom_line(aes(x = random_rank, y = mean_test_score, color = model_name), size=2) +
    geom_ribbon(aes(x=random_rank, ymin = min_test_score, ymax=max_test_score, fill=model_name), alpha=0.3) +
    geom_line(aes(x = random_rank, y = mean_test_score, color=model_name), size=2, data=res_datasets_bo_ %>% filter(random_rank <= 200), linetype="11") +
    #geom_text(aes(x=150, y=mean_test_score + 0.0007, color=model_name, label=str_c(model_name, " (Default)")), data=res_ %>% filter(random_rank == 0), size=7) +
    #geom_dl(aes(label = model_name, x=random_rank, y=mean_test_score, color=model_name), method = list(dl.combine("smart.grid"), cex=1.3))  +
    geom_text_repel(aes(label=model_name, 
                        color = model_name,
                        x = random_rank,
                        y =  mean_test_score),
                    data= (res_datasets_ %>% 
                             filter(random_rank == 1)),
                    bg.color='white', size = 6.5, bg.r=0.15,
                    nudge_y = 0., nudge_x = 0.3, min.segment.length=100)+
    coord_cartesian(ylim=c(y_inf, 1.0)) + 
    #scale_x_log10(limits=c(1, 700)) +
    scale_x_log10() +
    xlab("Number of random search iterations") +
    ylab(glue("Normalized test {score} of best  \n model (on valid set) up to this iteration")) +
    theme_minimal(base_size=22) +
    theme(legend.position="none") +
    colScale
}

# Numerical

benchmark_numerical <- read_csv("analyses/results/random_search_benchmark_numerical.csv")


# Classif

benchmark_bo <- read_csv("analyses/results/bo_nn.csv") %>% 
  bind_rows(read_csv("analyses/results/bo_trees.csv")) %>% 
  filter(!(data__keyword %in% (read_csv("analyses/results/bo_trees_2.csv") %>% select(data__keyword) %>% distinct())$data__keyword)) %>% 
  bind_rows(read_csv("analyses/results/bo_nn_2.csv")) %>% 
  bind_rows(read_csv("analyses/results/bo_trees_2.csv")) %>%
  rename() %>% 
  filter(data__categorical == F, data__regression == F) %>% 
  bind_rows(benchmark_numerical %>% 
               filter(benchmark == "numerical_classif_medium") %>% 
               filter(model_name %in% c("Resnet", "FT Transformer", "GradientBoostingTree", "XGBoost"))  %>% 
               filter(hp == "default") %>% 
               mutate(step = 0)) %>% 
  mutate(bo = T)

df <- benchmark_numerical %>% 
  filter(benchmark == "numerical_classif_medium") %>% 
  filter(model_name %in% c("Resnet", "FT Transformer", "GradientBoostingTree", "XGBoost")) %>% 
  mutate(bo = F)

df_total <- df %>% 
  bind_rows(benchmark_bo) %>% 
  filter(! data__keyword %in% c("california", "house_16H"))

plot_results_per_dataset_bo(df_total, score="accuracy", quantile=0.1, truncate_score=F, y_inf=0.5)


plot_aggregated_results_bo(df_total, score="accuracy", quantile=0.1, truncate_score=F, y_inf=0.5)

ggsave("analyses/plots/bo_classif_numerical.pdf", width=7, height=6, bg="white")


# Regression

benchmark_bo <- read_csv("analyses/results/bo_nn.csv") %>% 
  bind_rows(read_csv("analyses/results/bo_trees.csv")) %>% 
  filter(!(data__keyword %in% (read_csv("analyses/results/bo_trees_2.csv") %>% select(data__keyword) %>% distinct())$data__keyword)) %>% 
  bind_rows(read_csv("analyses/results/bo_nn_2.csv")) %>% 
  bind_rows(read_csv("analyses/results/bo_trees_2.csv")) %>%
  rename() %>% 
  filter(data__categorical == F, data__regression == T) %>% 
  mutate(mean_test_score = mean_r2_test, mean_val_score = mean_r2_val) %>% 
  bind_rows(benchmark_numerical %>% 
              filter(benchmark == "numerical_regression_medium") %>% 
              filter(model_name %in% c("Resnet", "FT Transformer", "GradientBoostingTree", "XGBoost"))  %>% 
              filter(hp == "default") %>% 
              mutate(step = 0)) %>% 
  mutate(bo = T)

df <- benchmark_numerical %>% 
  filter(benchmark == "numerical_regression_medium") %>% 
  filter(model_name %in% c("Resnet", "FT Transformer", "GradientBoostingTree", "XGBoost")) %>% 
  mutate(bo = F)

df_total <- df %>% 
  bind_rows(benchmark_bo) %>% 
  filter(data__keyword != "isolet")

plot_results_per_dataset_bo(df_total, score="R2 score", quantile=0.1, truncate_score=T, y_inf=0.5)


plot_aggregated_results_bo(df_total, score="R2 score", quantile=0.5, truncate_score=T, y_inf=0.3)

df_total %>% group_by(model_name, bo, data__keyword) %>% summarise(count = n()) %>% summarise(count = sum(count > 30))

ggsave("analyses/plots/bo_regression_numerical.pdf", width=7, height=6, bg="white")







