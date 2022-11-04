source("analyses/plot_utils.R")

df <- read_csv("launch_config/results/gbt_tabpfn.csv") %>% 
  bind_rows(read_csv("launch_config/results/tabpfn.csv"))

df_filtered <- df %>% 
  filter(! data__keyword %in% c(18, 469))
checks(df)
#View(df %>% group_by(model_name, data__keyword) %>% summarise(count = n()))

# Dataset by dataset
plot_results_per_dataset(df_filtered, "accuracy", truncate_scores = F, default_colscale = F, equalize_n_iteration = F)



#ggsave("analyses/plots/random_search_regression_numerical_datasets.pdf", width=15, height=10, bg="white")


# Aggregated

plot_aggregated_results(df_filtered, score="accuracy", quantile=0.1, y_inf=0.1, default_colscale = F, equalize_n_iteration = F)


#ggsave("analyses/plots/random_search_regression_numerical.pdf", width=16, height=9, bg="white")



df %>% 
  ggplot() +
  geom_point(aes(x = n_train, y = mean_test_score, color=model_name))



df_various_sizes <- read_csv("launch_config/results/tabpfn_various_sizes.csv") %>% 
  bind_rows(read_csv("launch_config/results/gbt_rf_various_sizes.csv"))

plot_results_per_dataset(df_various_sizes %>% filter(max_train_samples == 100), "accuracy", truncate_scores = F, default_colscale = F, equalize_n_iteration = F)

plot_aggregated_results(df_various_sizes %>% filter(max_train_samples == 1000), score="accuracy", quantile=0.1, y_inf=0.5, default_colscale = F, equalize_n_iteration = F)


