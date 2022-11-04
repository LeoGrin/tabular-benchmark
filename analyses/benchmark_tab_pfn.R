source("analyses/plot_utils.R")

df_numerical <- read_csv("launch_config/results/tab_pfn_default.csv") %>% 
  filter(data__categorical == 0) %>% 
  filter(!is.na(mean_test_score)) %>% 
  mutate(hp = "default")

datasets_numerical <- df_numerical$data__keyword

df_numerical <- df_numerical %>% 
  bind_rows(read_csv("launch_config/results/small_hgbt.csv") %>% 
              bind_rows(read_csv("launch_config/results/small_gbt.csv")) %>% 
               filter(data__categorical == 0) %>%
               filter(data__keyword %in% datasets_numerical) 
  )


checks(df_numerical)
#View(df %>% group_by(model_name, data__keyword) %>% summarise(count = n()))

# Dataset by dataset
plot_results_per_dataset(df_numerical, "accuracy", truncate_scores = F, default_colscale = F, equalize_n_iteration = F)


#ggsave("analyses/plots/random_search_regression_numerical_datasets.pdf", width=15, height=10, bg="white")


# Aggregated

plot_aggregated_results(df_numerical, score="accuracy", quantile=0.1, y_inf=0.1, default_colscale = F, equalize_n_iteration = F)


#ggsave("analyses/plots/random_search_regression_numerical.pdf", width=16, height=9, bg="white")



df_categorical <- read_csv("launch_config/results/tab_pfn_default.csv") %>% 
  filter(data__categorical == 1) %>% 
  filter(!is.na(mean_test_score)) %>% 
  mutate(hp = "default")

datasets_categorical <- df_categorical$data__keyword

df_categorical <- df_categorical %>% 
  bind_rows(read_csv("launch_config/results/small_hgbt.csv") %>% 
              bind_rows(read_csv("launch_config/results/small_gbt.csv")) %>% 
              filter(data__categorical == 1) %>%
              filter(data__keyword %in% datasets_categorical) 
  )


checks(df_categorical)
#View(df %>% group_by(model_name, data__keyword) %>% summarise(count = n()))

# Dataset by dataset
plot_results_per_dataset(df_categorical, "accuracy", truncate_scores = F, default_colscale = F, equalize_n_iteration = F)


#ggsave("analyses/plots/random_search_regression_numerical_datasets.pdf", width=15, height=10, bg="white")


# Aggregated

plot_aggregated_results(df_numerical, score="accuracy", quantile=0.1, y_inf=0.1, default_colscale = F, equalize_n_iteration = F)


