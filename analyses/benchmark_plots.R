source("analyses/random_search_utils.R")


######################################################
# Benchmark regression numerical medium

df <- read_csv("results/sweeps/sweeps_regression/benchmark_regression_random_medium_numerical.csv") %>% 
  select(model_name, data__keyword, mean_test_score, mean_val_score, mean_time) %>% 
  bind_rows(read_csv("results/sweeps/sweeps_classif/benchmark_regression_random_medium_numerical_no_transform.csv") %>% 
              mutate(model_name = "rf_no_transform") %>% 
              select(model_name, data__keyword, mean_test_score, mean_val_score, mean_time)) %>% 
  bind_rows(read_csv("results/sweeps/sweeps_classif/gbt_no_transformed.csv") %>% 
              mutate(model_name = "gbt_no_transform") %>% 
              select(model_name, data__keyword, mean_test_score, mean_val_score, mean_time) %>% 
              mutate(mean_test_score = as.numeric(mean_test_score),
                     mean_val_score = as.numeric(mean_val_score))) %>% 
  bind_rows(read_csv("results/sweeps/sweeps_classif/resnet_no_transformed.csv") %>% 
              mutate(model_name = "resnet_no_transform") %>% 
              select(model_name, data__keyword, mean_test_score, mean_val_score, mean_time) %>% 
              mutate(mean_test_score = as.numeric(mean_test_score),
                     mean_val_score = as.numeric(mean_val_score))) %>% 
  bind_rows(read_csv("results/sweeps/sweeps_classif/ft_no_transformed.csv") %>% 
              mutate(model_name = "ft_no_transform") %>% 
              select(model_name, data__keyword, mean_test_score, mean_val_score, mean_time) %>% 
              mutate(mean_test_score = as.numeric(mean_test_score),
                     mean_val_score = as.numeric(mean_val_score))) %>% 
  mutate(hp = "random") %>% 
  #bind_rows(read_csv("results/sweeps/sweeps_regression/benchmark_regression_default_medium_numerical.csv") %>% 
  #            mutate(hp="default")) %>% 
  filter(!is.na(mean_test_score), !is.na(mean_val_score)) %>% 
  mutate(mean_test_score = -mean_test_score,
         mean_val_score = -mean_val_score,
         mean_train_score = -mean_val_score)# %>% 


df <- bind_rows(read_csv("results/sweeps/sweeps_classif/benchmark_regression_random_medium_numerical_no_transform.csv") %>% 
              #mutate(model_name = "rf_no_transform") %>% 
              select(model_name, data__keyword, mean_test_score, mean_val_score, mean_time)) %>% 
  bind_rows(read_csv("results/sweeps/sweeps_classif/gbt_no_transformed.csv") %>% 
             # mutate(model_name = "gbt_no_transform") %>% 
              select(model_name, data__keyword, mean_test_score, mean_val_score, mean_time) %>% 
              mutate(mean_test_score = as.numeric(mean_test_score),
                     mean_val_score = as.numeric(mean_val_score))) %>% 
  bind_rows(read_csv("results/sweeps/sweeps_classif/resnet_no_transformed.csv") %>% 
              #mutate(model_name = "resnet_no_transform") %>% 
              select(model_name, data__keyword, mean_test_score, mean_val_score, mean_time) %>% 
              mutate(mean_test_score = as.numeric(mean_test_score),
                     mean_val_score = as.numeric(mean_val_score))) %>% 
  bind_rows(read_csv("results/sweeps/sweeps_classif/ft_no_transformed.csv") %>% 
             # mutate(model_name = "ft_no_transform") %>% 
              select(model_name, data__keyword, mean_test_score, mean_val_score, mean_time) %>% 
              mutate(mean_test_score = as.numeric(mean_test_score),
                     mean_val_score = as.numeric(mean_val_score))) %>% 
  mutate(hp = "random") %>% 
  #bind_rows(read_csv("results/sweeps/sweeps_regression/benchmark_regression_default_medium_numerical.csv") %>% 
  #            mutate(hp="default")) %>% 
  filter(!is.na(mean_test_score), !is.na(mean_val_score)) %>% 
  mutate(mean_test_score = -mean_test_score,
         mean_val_score = -mean_val_score,
         mean_train_score = -mean_val_score) %>% 
  rename()

df <- read_csv("results/sweeps/sweeps_regression/gbt_regression_new.csv") %>% 
  mutate(hp = "random") %>% 
  filter(!is.na(mean_test_score), !is.na(mean_val_score)) %>% 
  mutate(mean_test_score = -mean_test_score,
         mean_val_score = -mean_val_score,
         mean_train_score = -mean_val_score) %>% 
  rename()



res_datasets <- 
  df %>% 
  #filter(!is.na(mean_time)) %>% 
  normalize_no_variable(normalization_type = "quantile", quantile = 0.1) %>% 
  random_search_no_variable(n_shuffles=15, default_first = T)# %>% 
 # mutate(mean_test_score = if_else(mean_test_score < 0, 0, mean_test_score))

res_datasets %>% 
  filter(random_rank == 10) %>%
  ggplot() +
  geom_boxplot(aes(x = model_name, y=mean_test_score, color=model_name)) + #, width=0.1, height=0) +
  facet_wrap(~data__keyword, scales = "free") +
  theme_minimal(base_size=22) +
  #colScale +
  theme(legend.position="bottom", legend.title=element_blank(), legend.text = element_text(size=22))


res_datasets %>% 
  #filter(data__keyword != "year",
  #       random_rank > 40) %>% 
  group_by(random_rank, n_dataset, model_name) %>% 
  summarise(mean_test_score = mean(mean_test_score, na.rm=T)) %>% 
  ggplot() +
  geom_line(aes(x = random_rank, y = mean_test_score, color=model_name)) +
  scale_x_log10() +
  ylim(0.7, 1.0) +
  xlab("Number of random search iterations") +
  ylab("Normalized test score of best model \n (on valid set) up to this iteration") +
  theme_minimal(base_size=22) +
  colScale

library(directlabels)
res_datasets %>% 
  #filter(data__keyword != "year",
  #       random_rank > 40) %>% 
  group_by(random_rank, model_name, n_dataset) %>% 
  summarise(mean_test_score = mean(mean_test_score)) %>% 
  summarise(min_test_score = max(mean_test_score, na.rm=T),
            max_test_score = min(mean_test_score, na.rm=T),
            mean_test_score = mean(mean_test_score, na.rm=T)) %>% 
  ggplot() +
  #geom_hline(aes(yintercept=mean_test_score, color=model_name), data=res_ %>% filter(random_rank == 0), linetype="dotted", size=1.5, alpha=0.7) +
  geom_line(aes(x = random_rank, y = mean_test_score, color = model_name), size=2) +
  geom_ribbon(aes(x=random_rank, ymin = min_test_score, ymax=max_test_score, fill=model_name), alpha=0.3) +
  #geom_text(aes(x=150, y=mean_test_score + 0.0007, color=model_name, label=str_c(model_name, " (Default)")), data=res_ %>% filter(random_rank == 0), size=7) +
  geom_dl(aes(label = model_name, x=random_rank, y=mean_test_score, color=model_name), method = list(dl.combine("smart.grid"), cex=1.3))  +
  #ylim(0.6, 1.0) + 
  coord_cartesian(ylim=c(0.6, 1.0)) + 
  scale_x_log10() +
  xlab("Number of random search iterations") +
  ylab("Normalized test score of best model \n (on valid set) up to this iteration") +
  theme_minimal(base_size=22) +
  theme(legend.position="none") +
  colScale



res_datasets <- 
  df %>% 
  #filter(!is.na(mean_time)) %>% 
  normalize(variable=transformed_target, normalization_type = "quantile", quantile = 0.1) %>% 
  random_search(variable = transformed_target, n_shuffles=15, default_first = T)# %>% 
# mutate(mean_test_score = if_else(mean_test_score < 0, 0, mean_test_score))

library(directlabels)
res_datasets %>% 
  #filter(data__keyword != "year",
  #       random_rank > 40) %>% 
  group_by(random_rank, model_name, transformed_target, n_dataset) %>% 
  summarise(mean_test_score = mean(mean_test_score)) %>% 
  summarise(min_test_score = max(mean_test_score, na.rm=T),
            max_test_score = min(mean_test_score, na.rm=T),
            mean_test_score = mean(mean_test_score, na.rm=T)) %>% 
  ggplot() +
  #geom_hline(aes(yintercept=mean_test_score, color=model_name), data=res_ %>% filter(random_rank == 0), linetype="dotted", size=1.5, alpha=0.7) +
  geom_line(aes(x = random_rank, y = mean_test_score, color = transformed_target), size=2) +
  geom_ribbon(aes(x=random_rank, ymin = min_test_score, ymax=max_test_score, fill=transformed_target), alpha=0.3) +
  #geom_text(aes(x=150, y=mean_test_score + 0.0007, color=model_name, label=str_c(model_name, " (Default)")), data=res_ %>% filter(random_rank == 0), size=7) +
  #geom_dl(aes(label = model_name, x=random_rank, y=mean_test_score, color=model_name), method = list(dl.combine("smart.grid"), cex=1.3))  +
  #ylim(0.6, 1.0) + 
  coord_cartesian(ylim=c(0.6, 1.0)) + 
  scale_x_log10() +
  xlab("Number of random search iterations") +
  ylab("Normalized test score of best model \n (on valid set) up to this iteration") +
  theme_minimal(base_size=22) #+
 # theme(legend.position="none")# +
  #colScale

################################################################################


# Benchmark classification numerical medium

df <- read_csv("results/sweeps/sweeps_classif/benchmark/medium/gbt_classif.csv") %>%
  bind_rows(read_csv("results/sweeps/sweeps_classif/benchmark/medium/rf_classif.csv")) %>%  
  bind_rows(read_csv("results/sweeps/sweeps_classif/benchmark/medium/xgb_classif.csv")) %>%
  bind_rows(read_csv("results/sweeps/sweeps_classif/benchmark_classif_random_medium_checkpoint.csv") %>% 
              select(model_name, data__keyword, mean_val_score, mean_test_score, mean_time)) %>% 
  bind_rows(read_csv("results/sweeps/sweeps_classif/benchmark/medium/xgb_random_jannis.csv")) %>%
  bind_rows(read_csv("results/sweeps/sweeps_classif/benchmark/benchmark_classif_random_medium_numerical_bonus.csv")) %>% 
  bind_rows(read_csv("results/sweeps/sweeps_classif/benchmark/benchmark_classif_random_medium_bonus_2.csv")) %>% 
  mutate(hp="random") %>% 
  bind_rows(
    read_csv("results/sweeps/sweeps_classif/benchmark/medium/gbt_classif_default.csv") %>%
    bind_rows(read_csv("results/sweeps/sweeps_classif/benchmark/medium/rf_classif_default.csv")) %>% 
    bind_rows(read_csv("results/sweeps/sweeps_classif/benchmark/benchmark_classif_default_medium_numerical.csv")) %>% 
      mutate(hp="default")) %>% 
  select(model_name, data__keyword, mean_val_score, mean_test_score, mean_time, hp) %>% 
  filter(!is.na(mean_test_score), !is.na(mean_val_score), !is.na(model_name)) %>% 
  rename()

#res_datasets <- 
#  df %>% 
#  filter(!is.na(mean_time)) %>% 
#  normalize_no_variable(normalization_type = "quantile", quantile = 0.3) %>% 
#  random_search_no_variable(n_shuffles=15, default_first = T)


res_datasets <- 
  df %>% 
  filter(!is.na(mean_time)) %>% 
  normalize_no_variable(normalization_type = "quantile", quantile = 0.1) %>% 
  random_search_no_variable(n_shuffles=15, default_first = T)

res_datasets %>% 
  filter(random_rank == 30) %>%
  ggplot() +
  geom_boxplot(aes(x = model_name, y=mean_test_score, color=model_name)) + #, width=0.1, height=0) +
  facet_wrap(~data__keyword, scales = "free") +
  theme_minimal(base_size=22) +
  colScale +
  theme(legend.position="bottom", legend.title=element_blank(), legend.text = element_text(size=22))


res_datasets_ <- res_datasets %>% 
  group_by(random_rank, model_name, n_dataset) %>% 
  summarise(mean_test_score = mean(mean_test_score, na.rm=T)) %>% 
  summarise(min_test_score = min(mean_test_score, na.rm=T),#quantile(mean_test_score, 0.1, na.rm=T),
            max_test_score = max(mean_test_score, na.rm=T),#quantile(mean_test_score, 0.9, na.rm=T),
            mean_test_score = mean(mean_test_score, na.rm=T))

library(directlabels)

res_datasets_ %>% 
  ggplot() +
  geom_hline(aes(yintercept=mean_test_score, color=model_name), data=res_datasets_ %>% filter(random_rank == 0), linetype="dotted", size=1.5, alpha=0.7) +
  geom_line(aes(x = random_rank, y = mean_test_score, color = model_name), size=2) +
  geom_ribbon(aes(x=random_rank, ymin = min_test_score, ymax=max_test_score, fill=model_name), alpha=0.3) +
  #geom_text(aes(x=150, y=mean_test_score + 0.0007, color=model_name, label=str_c(model_name, " (Default)")), data=res_ %>% filter(random_rank == 0), size=7) +
  geom_dl(aes(label = model_name, x=random_rank, y=mean_test_score, color=model_name), method = list(dl.combine("smart.grid"), cex=1.3))  +
  coord_cartesian(ylim=c(0.5, 1.0)) + 
  scale_x_log10(limits=c(1, 1000)) +
  xlab("Number of random search iterations") +
  ylab("Normalized test score of best model \n (on valid set) up to this iteration") +
  theme_minimal(base_size=22) +
  theme(legend.position="none") +
  colScale

ggsave("analyses/plots/random_search_classif_numerical.jpg", width=7, height=6)


