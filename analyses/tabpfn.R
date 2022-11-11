source("analyses/plot_utils.R")

#df <- read_csv("launch_config/results/gbt_tabpfn.csv") %>% 
#  bind_rows(read_csv("launch_config/results/tabpfn.csv"))

df <- read_csv("launch_config/results/stacking_tabpfn_new.csv") %>% mutate(model_name = "stacking_tabpfn") %>% 
  #bind_rows(read_csv("launch_config/results/stacking_tabpfn.csv") %>% mutate(hp="random") %>% mutate(model_name = "stacking_tabpfn")) %>% 
  bind_rows(read_csv("launch_config/results/stacking_all.csv") %>% mutate(model_name = "stacking_all")) %>% 
  #bind_rows(read_csv("launch_config/results/stacking_all.csv") %>% mutate(hp = "random") %>% mutate(model_name = "stacking_all")) %>% 
  bind_rows(read_csv("launch_config/results/gbt_rf_various_sizes.csv")) %>% 
  bind_rows(read_csv("launch_config/results/tabpfn_various_sizes.csv")) %>%
  #bind_rows(read_csv("launch_config/results/log_reg.csv")) %>% 
  filter(max_train_samples == 1000) %>% 
  filter(data__categorical==F)

df_cc18 <- read_csv("launch_config/results/trees_cc18.csv") %>% 
  bind_rows(read_csv("launch_config/results/stacking_all_cc18.csv") %>% mutate(model_name = "stacking_all")) %>% 
  bind_rows(read_csv("launch_config/results/stacking_tabpfn_cc18.csv") %>% mutate(model_name = "stacking_tabpfn")) %>% 
  bind_rows(read_csv("launch_config/results/tabpfn.csv")) %>% 
  bind_rows(read_csv("launch_config/results/log_reg_cc18.csv")) %>% 
  filter(max_train_samples == 1000)

#df_filtered <- df %>% 
#  filter(! data__keyword %in% c(18, 469))
checks(df)
#View(df %>% group_by(model_name, data__keyword) %>% summarise(count = n()))

# Dataset by dataset
plot_results_per_dataset(df, "accuracy", truncate_scores = F, default_colscale = F, equalize_n_iteration = F)


#16, 54, 1462, 1480, 1494, 1510, 6332
# fourier features
# handcrafted image features

# plus other random stuff

#ggsave("analyses/plots/random_search_regression_numerical_datasets.pdf", width=15, height=10, bg="white")


# Aggregated

plot_aggregated_results(df_cc18, score="accuracy", quantile=0.1, y_inf=0.1, default_colscale = F, equalize_n_iteration = F)


#ggsave("analyses/plots/random_search_regression_numerical.pdf", width=16, height=9, bg="white")



df %>% 
  ggplot() +
  geom_point(aes(x = n_train, y = mean_test_score, color=model_name))



df_various_sizes <- read_csv("launch_config/results/tabpfn_various_sizes.csv") %>% 
  bind_rows(read_csv("launch_config/results/gbt_rf_various_sizes.csv"))

plot_results_per_dataset(df_various_sizes %>% filter(max_train_samples == 100), "accuracy", truncate_scores = F, default_colscale = F, equalize_n_iteration = F)

plot_aggregated_results(df_various_sizes %>% filter(max_train_samples == 1000), score="accuracy", quantile=0.1, y_inf=0.5, default_colscale = F, equalize_n_iteration = F)


