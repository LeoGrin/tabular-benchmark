source("analyses/plot_utils.R")

benchmark <- read_csv("analyses/results/benchmark_total.csv")

######################################################
# Benchmark regression numerical medium

df <- benchmark %>% 
  filter(benchmark == "numerical_regression_medium") %>% 
  filter(model_name != "HistGradientBoostingTree")

# checks
checks(df)

# Dataset by dataset

plot_results_per_dataset(df, "R2 score", truncate_scores = T)
ggsave("analyses/plots/random_search_regression_numerical_datasets.pdf", width=15, height=10, bg="white")


# Aggregated

plot_aggregated_results(df, y_inf=0.4, y_sup=0.95, score="R2 score", quantile=0.5, truncate_scores = T, text_size=8, theme_size=25, max_iter=400)
ggsave("analyses/plots/random_search_regression_numerical_poster.pdf", width=13.5, height=7, bg="white")

plot_aggregated_results(df, y_inf=0.5, y_sup=0.95, score="R2 score", quantile=0.5, truncate_scores = T, default_colscale = T)
ggsave("analyses/plots/random_search_regression_numerical.pdf", width=7, height=6, bg="white")


########################################################
# Benchmark classif numerical medium

df <- benchmark %>% 
  filter(benchmark == "numerical_classification_medium")

checks(df)

plot_results_per_dataset(df, "accuracy", default_colscale = T)
ggsave("analyses/plots/random_search_classif_numerical_datasets.pdf", width=15, height=10, bg="white")


# Aggregated

plot_aggregated_results(df, y_inf=0.55, y_sup=0.95, score="accuracy", quantile=0.1, truncate_scores = F, text_size=8, theme_size=25, max_iter=400)
ggsave("analyses/plots/random_search_classif_numerical_poster.pdf",  width=13.5, height=7, bg="white")

plot_aggregated_results(df, y_inf=0.5, y_sup=0.95, score="accuracy", quantile=0.1, truncate_scores = F)
ggsave("analyses/plots/random_search_classif_numerical.pdf", width=7, height=6, bg="white")
