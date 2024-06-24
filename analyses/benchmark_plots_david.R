library(dplyr)
library(stringr)
source("analyses/plot_utils.R")

#df <- read_csv("analyses/results/benchmark_cleaned_01_02_v2.csv")
df <- read_csv("analyses/results/grinsztajn_results.csv")

df <- df %>% 
  filter(model_name != "HistGradientBoostingTree")# %>% 
  #mutate(model_name = if_else(model_name == "david_not_simple",
  #                            "MLP-TD", model_name))



model_type <- function(row) {
  if (row["hp"] == "default") {
    if (str_detect(row["model_name"], "david")) {
      if (str_ends(row["model_name"], "_d") || 
          str_ends(row["model_name"], "_d_regressor") || 
          str_detect(row["model_name"], "tabr")) {
        return("D")
      }
      return("TD")
    } else {
      return("D")
    }
  } else {
    return("HPO")
  }
}

df <- df %>%
  mutate(
    model = str_replace_all(model_name, "_regressor", ""),
    model = str_replace_all(model, "_d", "")
  ) %>%
  mutate(model_name = recode(model,
                             "david_not_simple" = "RealMLP",
                             "MLP" = "MLP",
                             "Resnet" = "ResNet",
                             "RandomForest" = "RF",
                             "GradientBoostingTree" = "GBT",
                             "david_xgboost" = "XGB",
                             "david_lightgbm" = "LGBM",
                             "david_catboost" = "CatBoost",
                             "XGBoost" = "XGB",
                             "david_best" = "Best",
                             "david_best_hpo" = "Best",
                             "SAINT" = "SAINT",
                             "FT Transformer" = "FT-Transformer",
                             "david_tabrs" = "TabR-S"
  ))

df_classif <- df %>% 
  filter(benchmark == "numerical_classification_medium" | benchmark == "categorical_classification_medium")
# df_classif <- df_classif %>% 
#   mutate(model_name = case_when(
#     model_name == "david_catboost" ~ "CatBoost-TD",
#     model_name == "david_xgboost" ~ "XGBoost-TD",
#     model_name == "david_lightgbm" ~ "LGBM-TD",
#     TRUE ~ model_name))

checks(df_classif, 50)

plot_aggregated_results(df_classif, y_inf=0.6, y_sup=0.9, score="accuracy", quantile=0.1, truncate_scores = F, text_size=8, theme_size=25, max_iter=50,
                        default_colscale=F, equalize_n_iteration = F)


ggsave("analyses/plots/grinsztajn_up_to_classif.pdf", width=7, height=7, bg="white")


plot_results_per_dataset(df_classif, "accuracy", default_colscale = F, equalize_n_iteration = F) +
  ggtitle("Numerical classification")


df_reg <- df %>% 
  filter(benchmark == "numerical_regression_medium" | benchmark == "categorical_regression_medium")

#df_reg %>% select(benchmark) %>% distinct()

df_reg %>% 
  filter(model_name == "GBT", hp=="random") %>% 
  select(data__keyword) %>% 
  distinct(
  )

df_reg <- df_reg %>% 
  mutate(model_name = case_when(
    model_name == "david_catboost_regressor" ~ "CatBoost-TD",
    model_name == "david_xgboost_regressor" ~ "XGBoost-TD",
    model_name == "david_lightgbm_regressor" ~ "LGBM-TD",
    model_name == "david_not_simple_regressor" ~ "MLP-TD",
    TRUE ~ model_name))

plot_aggregated_results(df_reg, y_inf=0.4, y_sup=0.9, score="R2 score", quantile=0.5, truncate_scores = T, text_size=8, theme_size=25, max_iter=50,
                        default_colscale=F, equalize_n_iteration = F)

ggsave("analyses/plots/grinsztajn_up_to_regression.pdf", width=7, height=7, bg="white")


plot_aggregated_results(df_reg, y_inf=0.8, y_sup=1, score="accuracy", quantile=0.1, truncate_scores = T, text_size=8, theme_size=25, max_iter=50,
                        default_colscale=F, equalize_n_iteration = T)
