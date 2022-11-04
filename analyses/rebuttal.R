source("analyses/random_search_utils.R")
library(directlabels)
library(shadowtext)
library(ggrepel)
library(glue)

plot_results_per_dataset <- function(df_transformed, score="accuracy", truncate_scores=F, normalize=F, quantile=0.1, legend_size=22) {
  
  
  
  res_datasets <- df_transformed %>% 
    # Restrict to the min of iter for all model for each datasets
    left_join(df_transformed %>% group_by(data__keyword, model_name) %>% summarise(count = n()) %>% summarise(min_count = min(count)), by="data__keyword") %>% 
    filter(random_rank < min_count) %>% 
    ungroup() %>% 
    # Aggregate across shuffles
    group_by(random_rank, data__keyword, model_name) %>% 
    summarise(min_test_score = min(mean_test_score, na.rm=T),#quantile(mean_test_score, 0.1, na.rm=T),
              max_test_score = max(mean_test_score, na.rm=T),#quantile(mean_test_score, 0.9, na.rm=T),
              mean_test_score = mean(mean_test_score, na.rm=T)) %>% 
    ungroup() %>% 
    filter(min_test_score != Inf, max_test_score != -Inf) %>% 
    # prevent the ribbon from shifting the scale
    group_by(data__keyword) %>% 
    mutate(plot_min = min(mean_test_score, na.rm=T)) %>% # lowest line
    ungroup() %>% 
    rowwise() %>%
    mutate(min_test_score = max(min_test_score, plot_min)) %>% 
    ungroup()

  
  res_datasets %>% 
    ggplot() +
    geom_hline(aes(yintercept=mean_test_score, color=model_name), data=res_datasets %>% filter(random_rank == 0) %>% group_by(model_name, data__keyword) %>% summarise(mean_test_score = mean(mean_test_score, na.rm=T)), linetype="dotted", size=1.5, alpha=0.7) +
    geom_line(aes(x = random_rank, y = mean_test_score, color = model_name), size=2) +
    geom_ribbon(aes(x=random_rank, ymin = min_test_score, ymax=max_test_score, fill=model_name), alpha=0.3) +
    #geom_text(aes(x=150, y=mean_test_score + 0.0007, color=model_name, label=str_c(model_name, " (Default)")), data=res_ %>% filter(random_rank == 0), size=7) +
    #geom_dl(aes(label = model_name, x=random_rank, y=mean_test_score, color=model_name), method = list(dl.combine("smart.grid"), cex=1.3))  +
    facet_wrap(~data__keyword, scales="free") +
    #geom_blank(aes(y = y_min)) +
    coord_cartesian(ylim=c(res_datasets$y_min, res_datasets$y_max)) + 
    scale_x_log10() +#limits=c(1, 1000)) +
    xlab("Number of random search iterations") +
    ylab(str_interp(glue("Test {score} of best model \n (on valid set) up to this iteration"))) +
    theme_minimal(base_size=22) +
    theme(legend.position="bottom", legend.title=element_blank(), legend.text = element_text(size=legend_size))
}

plot_aggregated_results <- function(df_transformed, score="accuracy", quantile=0.1, truncate_scores=F, y_inf = 0.5) {
  
  res_datasets_ <- df_transformed %>% 
    group_by(random_rank, model_name, n_dataset) %>% 
    summarise(mean_test_score = mean(mean_test_score, na.rm=T)) %>% #summarise on datasets
    summarise(min_test_score = min(mean_test_score, na.rm=T),#quantile(mean_test_score, 0.1, na.rm=T),
              max_test_score = max(mean_test_score, na.rm=T),#quantile(mean_test_score, 0.9, na.rm=T),
              mean_test_score = mean(mean_test_score, na.rm=T))
  
  
  res_datasets_ %>% 
    ggplot() +
    geom_hline(aes(yintercept=mean_test_score, color=model_name), data=res_datasets_ %>% filter(random_rank == 0), linetype="dotted", size=1.5, alpha=0.7) +
    geom_line(aes(x = random_rank, y = mean_test_score, color = model_name), size=2) +
    geom_ribbon(aes(x=random_rank, ymin = min_test_score, ymax=max_test_score, fill=model_name), alpha=0.3) +
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
    theme(legend.position="none")
}

benchmark_numerical <- read_csv("analyses/results/random_search_benchmark_numerical.csv") %>% 
  filter(model_name == "GradientBoostingTree")# %>% 
  #filter(data__keyword %in% (read_csv("launch_config/results/resnet_several_random.csv") %>% select(data__keyword) %>% distinct())$data__keyword)

# Benchmark classif numerical medium
  
  
df <- benchmark_numerical %>% 
filter(benchmark == "numerical_classif_medium")

df_new <- read_csv("launch_config/results/gbt_several_random.csv") %>% 
  filter(data__keyword != "poker") %>% 
  #select(model_name, data__keyword, mean_val_score, mean_test_score, mean_time, hp) %>% 
  filter(!is.na(mean_test_score), !is.na(mean_val_score), !is.na(model_name)) %>% 
  rename() %>% 
  mutate(model_name = paste0(model_name, "_new")) %>% 
  group_by(sweep, data__keyword, model_name) %>% 
  mutate(random_rank=row_number(), n_dataset = sweep) %>% 
  ungroup() %>% 
  group_by(n_dataset) %>% 
  mutate(count = n()) %>% 
  filter(count > 1000) %>% 
  ungroup()

df %>% bind_rows(df_new) %>% 
  ggplot() +
  geom_jitter(aes(x=0, y = mean_test_score, color = n_dataset), alpha=0.3, width=3, height=0, size=1) +
  facet_wrap(~data__keyword, scales = "free")



df_new <- df_new %>% 
  group_by(n_dataset, model_name, data__keyword) %>% 
  mutate(best_val_score_so_far = cummax(mean_val_score)) %>% 
  mutate(mean_test_score = if_else(mean_val_score == best_val_score_so_far, mean_test_score, NA_real_)) %>% 
  # Only keep the scores for the first time we beat valid score, put the other to NA
  group_by(n_dataset, model_name, data__keyword, mean_val_score) %>% 
  mutate(mean_test_score = if_else(random_rank == min(random_rank), mean_test_score, NA_real_)) %>% 
  ungroup(mean_val_score) %>% 
  #for every dataset and {{variable}}, prolong the curve by the last best value when there is no value left
  ungroup() %>% 
  complete(n_dataset, model_name, data__keyword, random_rank) %>%
  group_by(n_dataset, model_name, data__keyword) %>% 
  mutate(mean_test_score = na.locf(mean_test_score, na.rm=F))%>% 
  ungroup()


res_datasets <- 
  df %>% 
  filter(!is.na(mean_time)) %>% 
  mutate_if(is.numeric, list(~na_if(., Inf))) %>% 
  mutate_if(is.numeric, list(~na_if(., -Inf)))



#normalize_no_variable(normalization_type = "quantile", quantile = 0.1) %>% 
df_transformed <- res_datasets %>% random_search_no_variable(n_shuffles=15, default_first = F) 

df_transformed <- df_transformed %>%
  mutate(n_dataset = as.character(n_dataset)) %>% 
  select(-transform__0__apply_on) %>% 
  left_join(read_csv("launch_config/results/gbt_several_random.csv") %>% 
                                                 filter(data__keyword != "poker") %>% 
                                                 #select(model_name, data__keyword, mean_val_score, mean_test_score, mean_time, hp) %>% 
                                                 filter(!is.na(mean_test_score), !is.na(mean_val_score), !is.na(model_name)) %>% 
                                                 rename() %>% 
                                                 group_by(sweep, data__keyword, model_name) %>% 
                                                 mutate(random_rank=row_number(), n_dataset = sweep) %>% 
                                                 ungroup() %>% 
                                                 group_by(n_dataset) %>% 
                                                 mutate(count = n()) %>% 
                                                 filter(count > 1000) %>% 
                                                 mutate(n_dataset = str_sub(n_dataset,-1,-1)) %>% 
                                                 ungroup() %>% group_by(model_name, data__keyword, n_dataset) %>% summarise(count = n()), by=c("model_name", "data__keyword", "n_dataset")) %>% 
  filter(random_rank < count)

# quantile = 0.1
# df %>% 
#   group_by(data__keyword) %>% 
#   mutate(mean_val_score = (mean_val_score - quantile(mean_val_score, quantile, na.rm=T)) / (max(mean_val_score, na.rm=T) - quantile(mean_val_score, quantile, na.rm=T)), 
#          mean_test_score = (mean_test_score - quantile(mean_test_score, quantile, na.rm=T)) / (max(mean_test_score, na.rm=T) - quantile(mean_test_score, quantile, na.rm=T))) %>% 
#   write_csv("analyses/benchmark_classif_numerical_medium_normalized.csv")

df <- df_new %>% 
  bind_rows(df_transformed %>% 
              mutate(n_dataset = as.character(n_dataset)))

checks(df)


df %>% group_by(model_name, data__keyword, n_dataset) %>% summarise(count = n())

plot_results_per_dataset(df, "accuracy")

ggsave("analyses/plots/different_runs_vs_bootstrap.jpg", width=8, height=10, bg="white")
#geom_blank(aes(y = 1))

quantile=0.1
truncate_scores = F



df <- benchmark_numerical %>% 
  filter(benchmark == "numerical_classif_medium")

df_new <- read_csv("launch_config/results/gbt_several_random.csv") %>% 
  filter(data__keyword != "poker") %>% 
  #select(model_name, data__keyword, mean_val_score, mean_test_score, mean_time, hp) %>% 
  filter(!is.na(mean_test_score), !is.na(mean_val_score), !is.na(model_name)) %>% 
  filter(!is.na(mean_time)) %>% 
  normalize_no_variable(normalization_type = "quantile", quantile =quantile) %>% 
  rename() %>% 
  mutate(model_name = paste0(model_name, "_new")) %>% 
  group_by(sweep, data__keyword, model_name) %>% 
  mutate(random_rank=row_number(), n_dataset = sweep) %>% 
  ungroup()

df_new <- df_new %>% 
  group_by(n_dataset, model_name, data__keyword) %>% 
  mutate(best_val_score_so_far = cummax(mean_val_score)) %>% 
  mutate(mean_test_score = if_else(mean_val_score == best_val_score_so_far, mean_test_score, NA_real_)) %>% 
  # Only keep the scores for the first time we beat valid score, put the other to NA
  group_by(n_dataset, model_name, data__keyword, mean_val_score) %>% 
  mutate(mean_test_score = if_else(random_rank == min(random_rank), mean_test_score, NA_real_)) %>% 
  ungroup(mean_val_score) %>% 
  #for every dataset and {{variable}}, prolong the curve by the last best value when there is no value left
  ungroup() %>% 
  complete(n_dataset, model_name, data__keyword, random_rank) %>%
  group_by(n_dataset, model_name, data__keyword) %>% 
  mutate(mean_test_score = na.locf(mean_test_score, na.rm=F))%>% 
  ungroup()


res_datasets <- 
  df %>% 
  filter(!is.na(mean_time)) %>% 
  mutate_if(is.numeric, list(~na_if(., Inf))) %>% 
  mutate_if(is.numeric, list(~na_if(., -Inf)))

res_datasets <- res_datasets %>% 
          normalize_no_variable(normalization_type = "quantile", quantile =quantile)




#normalize_no_variable(normalization_type = "quantile", quantile = 0.1) %>% 
df_transformed <- res_datasets %>% random_search_no_variable(n_shuffles=15, default_first = F) 



# quantile = 0.1
# df %>% 
#   group_by(data__keyword) %>% 
#   mutate(mean_val_score = (mean_val_score - quantile(mean_val_score, quantile, na.rm=T)) / (max(mean_val_score, na.rm=T) - quantile(mean_val_score, quantile, na.rm=T)), 
#          mean_test_score = (mean_test_score - quantile(mean_test_score, quantile, na.rm=T)) / (max(mean_test_score, na.rm=T) - quantile(mean_test_score, quantile, na.rm=T))) %>% 
#   write_csv("analyses/benchmark_classif_numerical_medium_normalized.csv")

df <- df_new %>% 
  bind_rows(df_transformed %>% 
              mutate(n_dataset = as.character(n_dataset)))

checks(df)


#View(df %>% group_by(model_name, data__keyword) %>% summarise(count = n()))

plot_results_per_dataset(df, "accuracy")


#if (normalize)
#  res_datasets <- res_datasets %>% normalize_no_variable(normalization_type = "quantile", quantile = quantile)
#if (truncate_scores)
#  res_datasets <- res_datasets %>% mutate(mean_test_score = if_else(mean_test_score < 0, 0, mean_test_score)) 


#normalize_no_variable(normalization_type = "quantile", quantile = 0.1) %>% 
#df_transformed <- res_datasets %>% random_search_no_variable(n_shuffles=15, default_first = T) 




#ggsave("analyses/plots/random_search_classif_numerical_datasets.jpg", width=15, height=10, bg="white")


# Aggregated

plot_aggregated_results(df, y_inf=0.6, score="accuracy", F)


#ggsave("analyses/plots/random_search_classif_numerical.jpg", width=7, height=6, bg="white")