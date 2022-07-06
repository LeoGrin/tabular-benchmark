source("analyses/random_search_utils.R")


######################################################
# Benchmark regression numerical medium


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
  bind_rows(read_csv("results/sweeps/sweeps_regression/resnet_regression.csv") %>% 
              select(model_name, data__keyword, mean_test_score, mean_val_score, mean_time, transformed_target) %>% 
              mutate_at(c("mean_test_score", "mean_val_score"), as.numeric, na.rm=T)) %>% 
  mutate(hp = "random") %>% 
  filter(!is.na(mean_test_score), !is.na(mean_val_score)) %>% 
  mutate(mean_test_score = -mean_test_score,
         mean_val_score = -mean_val_score,
         mean_train_score = -mean_val_score) %>% 
  select(model_name, data__keyword, mean_test_score, mean_val_score, mean_time, hp, transformed_target) %>% 
  rename()


df <- read_csv("results/sweeps/sweeps_regression/benchmark_regression_random_medium_2.csv") %>% 
  mutate(mean_r2_test = as.numeric(mean_r2_test),
         mean_r2_val = as.numeric(mean_r2_val)) %>% 
  select(model_name, data__keyword, mean_r2_test, mean_r2_val, mean_time) %>% 
  bind_rows(read_csv("results/sweeps/sweeps_regression/saint_regression_4.csv") %>%
              bind_rows(read_csv("results/sweeps/sweeps_regression/saint_regression_3.csv")) %>% 
              #filter(model__params__dim < 256) %>% 
              select(model_name, data__keyword, mean_r2_test, mean_r2_val, mean_time) %>%
              mutate(model_name = "SAINT")) %>% 
  mutate(hp="random") %>% 
  bind_rows(read_csv("results/sweeps/sweeps_regression/benchmark_regression_default_medium_2.csv") %>% 
            bind_rows(read_csv("results/sweeps/sweeps_regression/saint_regression_default.csv") %>%
                        bind_rows(read_csv("results/sweeps/sweeps_regression/saint_regression_default_2.csv")) %>% 
                        distinct(data__keyword, .keep_all = T) %>% 
                        mutate(model_name = "SAINT")) %>% 
              mutate(hp="default") %>% 
              select(model_name, data__keyword, mean_r2_test, mean_r2_val, mean_time, hp)) %>% 
  filter(!is.na(mean_r2_test), !is.na(mean_r2_val), !is.na(data__keyword)) %>% 
  #filter(transformed_target==F) %>% 
  mutate(mean_test_score = mean_r2_test,
         mean_val_score = mean_r2_val) %>% 
  rename()
  #rename()

View(df %>% group_by(model_name, data__keyword) %>% summarise(count = n()))

quantile <- 0.6

df <- read_csv("results/sweeps/sweeps_regression/benchmark_regression_random_medium_2.csv") %>% 
  mutate(hp="random") %>% 
  mutate(mean_r2_test = as.numeric(mean_r2_test),
         mean_r2_val = as.numeric(mean_r2_val)) %>% 
  #select(model_name, data__keyword, mean_r2_test, mean_r2_val, mean_time, hp) %>% 
  filter(!is.na(mean_r2_test), !is.na(mean_r2_val), !is.na(data__keyword)) %>% 
  #filter(transformed_target==F) %>% 
  mutate(mean_test_score = mean_r2_test,
         mean_val_score = mean_r2_val) %>% 
  #rename() %>% 
  mutate(mean_val_score = (mean_val_score - quantile(mean_val_score, quantile, na.rm=T)) / (max(mean_val_score, na.rm=T) - quantile(mean_val_score, quantile, na.rm=T)), 
         mean_test_score = (mean_test_score - quantile(mean_test_score, quantile, na.rm=T)) / (max(mean_test_score, na.rm=T) - quantile(mean_test_score, quantile, na.rm=T)))# %>% 

df %>% write_csv("cleaned_regression_numerical_medium.csv")

df %>% group_by(model_name) %>% summarise(count = n(
))

df <- read_csv("results/sweeps/sweeps_regression/benchmark_regression_random_medium_2.csv") %>% 
  mutate(hp="random") %>% 
  mutate(mean_r2_test = as.numeric(mean_r2_test),
         mean_r2_val = as.numeric(mean_r2_val)) %>% 
  #select(model_name, data__keyword, mean_r2_test, mean_r2_val, mean_time, hp) %>% 
  filter(!is.na(mean_r2_test), !is.na(mean_r2_val), !is.na(data__keyword))

df %>% 
  mutate(n_iter = case_when(
    n_test > 6000 ~ 1,
    n_test > 3000 ~ 2,
    n_test > 1000 ~ 3,
    TRUE ~5
  )) %>% 
  summarise(s =  sum(mean_time * n_iter) / 3600)

############################################
# Benchmark regression categorical medium

df <- read_csv("results/sweeps/sweeps_regression/resnet_regression_categorical.csv") %>% 
  select(model_name, data__keyword, mean_r2_test, mean_r2_val, mean_time) %>% 
  mutate(mean_r2_test = as.numeric(mean_r2_test),
         mean_r2_val = as.numeric(mean_r2_val)) %>% 
  bind_rows(read_csv("results/sweeps/sweeps_regression/hgbt_regression_categorical.csv") %>% 
              select(model_name, data__keyword, mean_r2_test, mean_r2_val, mean_time) %>% 
              mutate(mean_r2_test = as.numeric(mean_r2_test),
                     mean_r2_val = as.numeric(mean_r2_val))) %>% 
  bind_rows(read_csv("results/sweeps/sweeps_regression/ft_transformer_regression_categorical.csv")) %>% 
  mutate(hp="random") %>% 
  filter(!is.na(mean_r2_test), !is.na(mean_r2_val), !is.na(data__keyword)) %>% 
  #filter(transformed_target==F) %>% 
  mutate(mean_test_score = mean_r2_test,
         mean_val_score = mean_r2_val) #%>% 
  #filter(!(data__keyword %in% c("Allstate_Claims_Severity", "LoanDefaultPrediction", "Mercedes_Benz_Greener_Manufacturing")))

df <-read_csv("results/sweeps/sweeps_regression/benchmark_categorical_regression_medium.csv") %>% 
  select(model_name, data__keyword, mean_r2_test, mean_r2_val, mean_time) %>% 
  mutate(mean_r2_test = as.numeric(mean_r2_test),
         mean_r2_val = as.numeric(mean_r2_val)) %>% 
  bind_rows(read_csv("results/sweeps/sweeps_regression/xgb_regression_categorical.csv")) %>% 
  bind_rows(read_csv("results/sweeps/sweeps_regression/rf_regression_categorical_bonus.csv")) %>% 
  mutate(hp = "random") %>% 
  bind_rows(read_csv("results/sweeps/sweeps_regression/benchmark_categorical_regression_medium_default.csv") %>% 
              select(model_name, data__keyword, mean_r2_test, mean_r2_val, mean_time) %>% 
              mutate(mean_r2_test = as.numeric(mean_r2_test),
                     mean_r2_val = as.numeric(mean_r2_val)) %>%
              bind_rows(read_csv("results/sweeps/sweeps_regression/xgb_regression_categorical_default.csv")) %>% 
              mutate(hp = "default")) %>% 
  filter(!is.na(mean_r2_test), !is.na(mean_r2_val), !is.na(data__keyword)) %>% 
  mutate(mean_test_score = mean_r2_test,
         mean_val_score = mean_r2_val) %>% 
  rename()

df <-read_csv("results/sweeps/sweeps_regression/resnet_categorical_regression_large.csv") %>% 
  select(model_name, data__keyword, mean_r2_test, mean_r2_val, mean_time) %>% 
  mutate(mean_r2_test = as.numeric(mean_r2_test),
         mean_r2_val = as.numeric(mean_r2_val)) %>% 
  bind_rows(read_csv("results/sweeps/sweeps_regression/ft_categorical_regression_large.csv") %>% 
              select(model_name, data__keyword, mean_r2_test, mean_r2_val, mean_time) %>% 
              mutate(mean_r2_test = as.numeric(mean_r2_test),
                     mean_r2_val = as.numeric(mean_r2_val))) %>% 
  #bind_rows(read_csv("results/sweeps/sweeps_regression/hgbt_categorical_regression_large.csv") %>% 
  #            select(model_name, data__keyword, mean_r2_test, mean_r2_val, mean_time) %>% 
  #            mutate(mean_r2_test = as.numeric(mean_r2_test),
  #                   mean_r2_val = as.numeric(mean_r2_val))) %>% 
  bind_rows(read_csv("results/sweeps/sweeps_regression/gbt_categorical_regression_large.csv") %>% 
              select(model_name, data__keyword, mean_r2_test, mean_r2_val, mean_time) %>% 
              mutate(mean_r2_test = as.numeric(mean_r2_test),
                     mean_r2_val = as.numeric(mean_r2_val))) %>% 
  bind_rows(read_csv("results/sweeps/sweeps_regression/large/hgbt_regression_categorical_large.csv") %>% 
              select(model_name, data__keyword, mean_r2_test, mean_r2_val, mean_time) %>% 
              mutate(mean_r2_test = as.numeric(mean_r2_test),
                     mean_r2_val = as.numeric(mean_r2_val))) %>% 
  bind_rows(read_csv("results/sweeps/sweeps_regression/large/xgb_regression_categorical_large.csv") %>% 
              select(model_name, data__keyword, mean_r2_test, mean_r2_val, mean_time) %>% 
              mutate(mean_r2_test = as.numeric(mean_r2_test),
                     mean_r2_val = as.numeric(mean_r2_val))) %>% 
  bind_rows(read_csv("results/sweeps/sweeps_regression/large/saint_regression_categorical_large.csv") %>% 
              select(model_name, data__keyword, mean_r2_test, mean_r2_val, mean_time) %>% 
              mutate(mean_r2_test = as.numeric(mean_r2_test),
                     mean_r2_val = as.numeric(mean_r2_val)) %>% 
              mutate(model_name = "SAINT")) %>% 
  mutate(hp = "random") %>% 
  filter(!is.na(mean_r2_test), !is.na(mean_r2_val), !is.na(data__keyword)) %>% 
  mutate(mean_test_score = mean_r2_test,
         mean_val_score = mean_r2_val) %>% 
  rename() %>% 
  filter(data__keyword != "Allstate_Claims_Severity",
         data__keyword != "LoanDefaultPrediction")


df %>% 
  mutate(n_iter = case_when(
    n_test > 6000 ~ 1,
    n_test > 3000 ~ 2,
    n_test > 1000 ~ 3,
    TRUE ~5
  )) %>% 
  summarise(s =  sum(mean_time * n_iter) / 3600)


res_datasets <- 
  df %>% 
  #filter(!is.na(mean_time)) %>% 
  normalize_no_variable(normalization_type = "quantile", quantile = 0.5) %>% 
  mutate(mean_test_score = if_else(mean_test_score < 0, 0, mean_test_score)) %>% 
  random_search_no_variable(n_shuffles=15, default_first = T)# %>% 
 # mutate(mean_test_score = if_else(mean_test_score < 0, 0, mean_test_score))


res_datasets %>% 
  #filter(!(data__keyword %in% c("Allstate_Claims_Severity", "LoanDefaultPrediction"))) %>% 
  group_by(random_rank, model_name, data__keyword) %>% 
  summarise(min_test_score = min(mean_test_score, na.rm=T),#quantile(mean_test_score, 0.1, na.rm=T),
            max_test_score = max(mean_test_score, na.rm=T),#quantile(mean_test_score, 0.9, na.rm=T),
            mean_test_score = mean(mean_test_score, na.rm=T)) %>% 
  ggplot() +
  #geom_hline(aes(yintercept=mean_test_score, color=model_name), data=res_datasets_ %>% filter(random_rank == 0), linetype="dotted", size=1.5, alpha=0.7) +
  geom_line(aes(x = random_rank, y = mean_test_score, color = model_name), size=2, alpha=1) +
  geom_ribbon(aes(x=random_rank, ymin = min_test_score, ymax=max_test_score, fill=model_name), alpha=0.3) +
  #geom_text(aes(x=150, y=mean_test_score + 0.0007, color=model_name, label=str_c(model_name, " (Default)")), data=res_ %>% filter(random_rank == 0), size=7) +
  #geom_dl(aes(label = model_name, x=random_rank, y=mean_test_score, color=model_name), method = list(dl.combine("smart.grid"), cex=1.3))  +
  facet_wrap(~data__keyword, scales="free") +
  #coord_cartesian(ylim=c(0.5, 1.0)) + 
  scale_x_log10(limits=c(1, 1000)) +
  xlab("Number of random search iterations") +
  ylab("Normalized test score of best model \n (on valid set) up to this iteration") +
  theme_minimal(base_size=22)# +
  #theme(legend.position="none") +
  #colScale

res_datasets_ <- res_datasets %>% 
  #filter(data__keyword != "isolet") %>% 
  filter(!(data__keyword %in% c("Allstate_Claims_Severity", "LoanDefaultPrediction"))) %>% 
  group_by(random_rank, model_name, n_dataset) %>% 
  summarise(mean_test_score = mean(mean_test_score, na.rm=T)) %>% #summarise on <- datasets
  summarise(min_test_score = min(mean_test_score),#quantile(mean_test_score, 0.1, na.rm=T),
            max_test_score = max(mean_test_score),#quantile(mean_test_score, 0.9, na.rm=T),
            mean_test_score = mean(mean_test_score))

library(directlabels)
library(shadowtext)
library(ggrepel)

#res_datasets_ <- res_datasets_ %>% mutate(model_name = fct_relevel(model_name, c("RandomForest", "GradientBoostingTree", "FT Transformer", "Resnet")))

res_datasets_ %>% 
  ggplot() +
  geom_hline(aes(yintercept=mean_test_score, color=model_name), data=res_datasets_ %>% filter(random_rank == 0), linetype="dotted", size=1.5, alpha=0.7) +
  geom_line(aes(x = random_rank, y = mean_test_score, color = model_name), size=2) +
  geom_ribbon(aes(x=random_rank, ymin = min_test_score, ymax=max_test_score, fill=model_name), alpha=0.3) +
  #geom_text(aes(x=150, y=mean_test_score + 0.0007, color=model_name, label=str_c(model_name, " (Default)")), data=res_ %>% filter(random_rank == 0), size=7) +
  #
  #geom_dl(aes(label = model_name, x=random_rank, y=mean_test_score, color=model_name), 
  #        method = list(dl.trans(x = x - 5), "smart.grid", cex=1.5, bg.color="white"))  +
  geom_text_repel(aes(label=model_name, 
                      color = model_name,
                      x = random_rank,
                      y =  mean_test_score),
                data= (res_datasets_ %>% 
                           filter(random_rank == 1)),
                 bg.color='white', size = 6.5, bg.r=0.15,
                nudge_y = 0.03, nudge_x = 0.1, min.segment.length=10)+
  #coord_cartesian(ylim=c(0.5, 1.0)) + 
  scale_x_log10(limits=c(1, 550)) +
  xlab("Number of random search iterations") +
  ylab("Normalized R2 test score of best \n model (on valid set) up to this iteration") +
  theme_minimal(base_size=22) +
  theme(legend.position="none") +
  colScale

ggsave("analyses/plots/random_search_regression_categorical.jpg", width=7, height=6)



#####################
# Benchmark regression categorical large

df <- read_csv("results/sweeps/sweeps_regression/resnet_regression_categorical_large.csv") %>% 
  select(model_name, data__keyword, mean_r2_test, mean_r2_val, mean_time) %>% 
  mutate(mean_r2_test = as.numeric(mean_r2_test),
         mean_r2_val = as.numeric(mean_r2_val)) %>% 
  bind_rows(read_csv("results/sweeps/sweeps_regression/hgbt_regression_categorical_large.csv") %>% 
              select(model_name, data__keyword, mean_r2_test, mean_r2_val, mean_time) %>% 
              mutate(mean_r2_test = as.numeric(mean_r2_test),
                     mean_r2_val = as.numeric(mean_r2_val))) %>% 
  bind_rows(read_csv("results/sweeps/sweeps_regression/ft_regression_categorical_large.csv")) %>% 
  
  mutate(hp="random") %>% 
  filter(!is.na(mean_r2_test), !is.na(mean_r2_val), !is.na(data__keyword)) %>% 
  #filter(transformed_target==F) %>% 
  mutate(mean_test_score = mean_r2_test,
         mean_val_score = mean_r2_val) %>% 
  filter(!(data__keyword %in% c("Allstate_Claims_Severity")))
  #filter(!(data__keyword %in% c("Allstate_Claims_Severity", "LoanDefaultPrediction", "Mercedes_Benz_Greener_Manufacturing")))

df %>% 
  mutate(n_iter = case_when(
    n_test > 6000 ~ 1,
    n_test > 3000 ~ 2,
    n_test > 1000 ~ 3,
    TRUE ~5
  )) %>% 
  summarise(s =  sum(mean_time * n_iter) / 3600)

res_datasets <- 
  df %>% 
  filter(!is.na(mean_time)) %>% 
  normalize_no_variable(normalization_type = "quantile", quantile = 0.7) %>% 
  mutate(mean_test_score = if_else(mean_test_score < 0, 0, mean_test_score)) %>% 
  random_search_no_variable(n_shuffles=15, default_first = T)# %>% 
# mutate(mean_test_score = if_else(mean_test_score < 0, 0, mean_test_score))


res_datasets %>% 
  group_by(random_rank, model_name, data__keyword) %>% 
  summarise(min_test_score = min(mean_test_score, na.rm=T),#quantile(mean_test_score, 0.1, na.rm=T),
            max_test_score = max(mean_test_score, na.rm=T),#quantile(mean_test_score, 0.9, na.rm=T),
            mean_test_score = mean(mean_test_score, na.rm=T)) %>% 
  ggplot() +
  #geom_hline(aes(yintercept=mean_test_score, color=model_name), data=res_datasets_ %>% filter(random_rank == 0), linetype="dotted", size=1.5, alpha=0.7) +
  geom_line(aes(x = random_rank, y = mean_test_score, color = model_name), size=2) +
  geom_ribbon(aes(x=random_rank, ymin = min_test_score, ymax=max_test_score, fill=model_name), alpha=0.3) +
  #geom_text(aes(x=150, y=mean_test_score + 0.0007, color=model_name, label=str_c(model_name, " (Default)")), data=res_ %>% filter(random_rank == 0), size=7) +
  #geom_dl(aes(label = model_name, x=random_rank, y=mean_test_score, color=model_name), method = list(dl.combine("smart.grid"), cex=1.3))  +
  facet_wrap(~data__keyword) +
  coord_cartesian(ylim=c(0., 1.0)) + 
  scale_x_log10(limits=c(1, 1000)) +
  xlab("Number of random search iterations") +
  ylab("Normalized test score of best model \n (on valid set) up to this iteration") +
  theme_minimal(base_size=22)# +
#theme(legend.position="none") +
#colScale

res_datasets_ <- res_datasets %>% 
  group_by(random_rank, model_name, n_dataset) %>% 
  summarise(mean_test_score = mean(mean_test_score, na.rm=T)) %>% #summarise on datasets
  summarise(min_test_score = min(mean_test_score, na.rm=T),#quantile(mean_test_score, 0.1, na.rm=T),
            max_test_score = max(mean_test_score, na.rm=T),#quantile(mean_test_score, 0.9, na.rm=T),
            mean_test_score = mean(mean_test_score, na.rm=T))

library(directlabels)
library(shadowtext)
library(ggrepel)

res_datasets_ %>% 
  ggplot() +
  geom_hline(aes(yintercept=mean_test_score, color=model_name), data=res_datasets_ %>% filter(random_rank == 0), linetype="dotted", size=1.5, alpha=0.7) +
  geom_line(aes(x = random_rank, y = mean_test_score, color = model_name), size=2) +
  geom_ribbon(aes(x=random_rank, ymin = min_test_score, ymax=max_test_score, fill=model_name), alpha=0.3) +
  #geom_text(aes(x=150, y=mean_test_score + 0.0007, color=model_name, label=str_c(model_name, " (Default)")), data=res_ %>% filter(random_rank == 0), size=7) +
  #
  #geom_dl(aes(label = model_name, x=random_rank, y=mean_test_score, color=model_name), 
  #        method = list(dl.trans(x = x - 5), "smart.grid", cex=1.5, bg.color="white"))  +
  geom_text_repel(aes(label=model_name, 
                      color = model_name,
                      x = random_rank,
                      y =  mean_test_score),
                  data= (res_datasets_ %>% 
                           filter(random_rank == 1)),
                  bg.color='white', size = 6,
                  nudge_y = 0.04, nudge_x = 0.1, min.segment.length=10)+
  #coord_cartesian(ylim=c(0.25, 1.0)) + 
  scale_x_log10(limits=c(1, 400)) +
  xlab("Number of random search iterations") +
  ylab("Normalized test score of best model \n (on valid set) up to this iteration") +
  theme_minimal(base_size=22) +
  theme(legend.position="none")# +
#colScale

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
  normalize(variable=transformed_target, normalization_type = "quantile", quantile = 0.6) %>% 
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
  geom_line(aes(x = random_rank, y = mean_test_score, color = transformed_target, linetype=model_name), size=2) +
  #geom_ribbon(aes(x=random_rank, ymin = min_test_score, ymax=max_test_score, fill=transformed_target, group=model_name), alpha=0.3) +
  #geom_text(aes(x=150, y=mean_test_score + 0.0007, color=model_name, label=str_c(model_name, " (Default)")), data=res_ %>% filter(random_rank == 0), size=7) +
  #geom_dl(aes(label = model_name, x=random_rank, y=mean_test_score, color=model_name), method = list(dl.combine("smart.grid"), cex=1.3))  +
  #ylim(0.6, 1.0) + 
  coord_cartesian(ylim=c(0.6, 1.0)) + 
  scale_x_log10() +
  xlab("Number of random search iterations") +
  ylab("Normalized test score of best model \n (on valid set) up to this iteration") +
  theme_minimal(base_size=22) +
 # theme(legend.position="none")# +
  colScale

res_datasets %>% 
  #filter(data__keyword != "year",
  #       random_rank > 40) %>% 
  group_by(random_rank, model_name, transformed_target, data__keyword) %>% 
  summarise(min_test_score = max(mean_test_score, na.rm=T),
            max_test_score = min(mean_test_score, na.rm=T),
            mean_test_score = mean(mean_test_score, na.rm=T)) %>% 
  ggplot() +
  #geom_hline(aes(yintercept=mean_test_score, color=model_name), data=res_ %>% filter(random_rank == 0), linetype="dotted", size=1.5, alpha=0.7) +
  geom_line(aes(x = random_rank, y = mean_test_score, color = transformed_target, linetype=model_name), size=2) +
  #geom_ribbon(aes(x=random_rank, ymin = min_test_score, ymax=max_test_score, fill=model_name), alpha=0.3) +
  facet_wrap(~data__keyword) +
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
  bind_rows(read_csv("results/sweeps/sweeps_classif/saint_classif.csv") %>% mutate(model_name = "SAINT")) %>% 
  #bind_rows(read_csv("results/sweeps/sweeps_classif/gbt_new_classif_numeric.csv") %>% 
  #            mutate(model_name = "gbt_new")) %>%  
  mutate(hp="random") %>% 
  #bind_rows(read_csv("results/sweeps/sweeps_classif/benchmark/medium/gbt_classif_default.csv") %>% 
  #            mutate(model_name = "gbt_new") %>% 
  #            mutate(hp = "default")) %>% 
  bind_rows(read_csv("results/sweeps/sweeps_classif/benchmark/medium/gbt_classif_default.csv") %>%
    bind_rows(read_csv("results/sweeps/sweeps_classif/benchmark/medium/rf_classif_default.csv")) %>% 
    bind_rows(read_csv("results/sweeps/sweeps_classif/benchmark/benchmark_classif_default_medium_numerical.csv")) %>%
    bind_rows(read_csv("results/sweeps/sweeps_classif/saint_classif_default.csv") %>% 
                mutate(model_name = "SAINT")) %>% 
      mutate(hp="default")) %>% 
  filter(data__keyword != "poker") %>% 
  select(model_name, data__keyword, mean_val_score, mean_test_score, mean_time, hp) %>% 
  filter(!is.na(mean_test_score), !is.na(mean_val_score), !is.na(model_name)) %>% 
  rename() #%>% 
  #filter(!is.na(model__max_depth))

quantile <- 0.1
df <- read_csv("results/sweeps/sweeps_classif/benchmark/medium/gbt_classif.csv") %>%
  bind_rows(read_csv("results/sweeps/sweeps_classif/benchmark/medium/rf_classif.csv")) %>%  
  bind_rows(read_csv("results/sweeps/sweeps_classif/benchmark/medium/xgb_classif.csv")) %>%
  bind_rows(read_csv("results/sweeps/sweeps_classif/benchmark_classif_random_medium_checkpoint.csv")) %>% 
  bind_rows(read_csv("results/sweeps/sweeps_classif/benchmark/medium/xgb_random_jannis.csv")) %>%
  bind_rows(read_csv("results/sweeps/sweeps_classif/benchmark/benchmark_classif_random_medium_numerical_bonus.csv")) %>% 
  bind_rows(read_csv("results/sweeps/sweeps_classif/benchmark/benchmark_classif_random_medium_bonus_2.csv")) %>% 
  #bind_rows(read_csv("results/sweeps/sweeps_classif/gbt_new_classif_numeric.csv") %>% 
  #            mutate(model_name = "gbt_new")) %>%  
  #bind_rows(read_csv("results/sweeps/sweeps_classif/benchmark/medium/gbt_classif_default.csv") %>% 
  #            mutate(model_name = "gbt_new") %>% 
  #            mutate(hp = "default")) %>% 
  bind_rows(read_csv("results/sweeps/sweeps_classif/benchmark/medium/gbt_classif_default.csv") %>%
               bind_rows(read_csv("results/sweeps/sweeps_classif/benchmark/medium/rf_classif_default.csv")) %>% 
               bind_rows(read_csv("results/sweeps/sweeps_classif/benchmark/benchmark_classif_default_medium_numerical.csv")) %>% 
               mutate(hp="default")) %>% 
  filter(data__keyword != "poker") %>% 
  filter(!is.na(mean_test_score), !is.na(mean_val_score), !is.na(model_name)) %>% 
  rename()
  #group_by(data__keyword)# %>% 
  #mutate(mean_val_score = (mean_val_score - quantile(mean_val_score, quantile, na.rm=T)) / (max(mean_val_score, na.rm=T) - quantile(mean_val_score, quantile, na.rm=T)), 
  #       mean_test_score = (mean_test_score - quantile(mean_test_score, quantile, na.rm=T)) / (max(mean_test_score, na.rm=T) - quantile(mean_test_score, quantile, na.rm=T)))

df %>% 
  ungroup() %>% 
  mutate(n_iter = case_when(
    n_test > 6000 ~ 1,
    n_test > 3000 ~ 2,
    n_test > 1000 ~ 3,
    TRUE ~5
  )) %>% 
  summarise(s =  sum(mean_time * n_iter) / 3600)

df %>% write_csv("results/sweeps/cleaned_classification_numerical_medium.csv")

View(df %>% select(model_name, data__keyword, mean_test_score))

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
  #colScale +
  theme(legend.position="bottom", legend.title=element_blank(), legend.text = element_text(size=22))


View(df %>% 
  group_by(model_name, data__keyword) %>% 
  summarise(m = mean(mean_test_score, na.rm=T), max = max(mean_test_score, na.rm=T), min = min(mean_test_score, na.rm=T)))

res_datasets %>% 
  group_by(model_name, random_rank, data__keyword) %>% 
  summarise(min_test_score = min(mean_test_score, na.rm=T),
            max_test_score = max(mean_test_score, na.rm=T),
            mean_test_score = mean(mean_test_score, na.rm=T)) %>% 
  ggplot() +
  geom_ribbon(aes(x = random_rank, ymax= max_test_score, ymin=min_test_score, fill=model_name), alpha=0.2)+
  geom_line(aes(x = random_rank, y = mean_test_score, color=model_name), size=2) +
  scale_x_log10()+
  facet_wrap(~data__keyword, scales="free")

res_datasets_ <- res_datasets %>% 
  group_by(random_rank, model_name, n_dataset) %>% 
  summarise(mean_test_score = mean(mean_test_score)) %>% 
  summarise(min_test_score = min(mean_test_score),#quantile(mean_test_score, 0.1, na.rm=T),
            max_test_score = max(mean_test_score),#quantile(mean_test_score, 0.9, na.rm=T),
            mean_test_score = mean(mean_test_score))

library(directlabels)
library(ggrepel)
library(shadowtext)


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
                           filter(random_rank == 0)),
                  bg.color='white', size = 6.5, bg.r=0.15,
                  nudge_y = 0.03, nudge_x = 0.1, min.segment.length=100)+
  coord_cartesian(ylim=c(0.6, 1)) + 
  scale_x_log10(limits=c(1, 550)) +
  xlab("Number of random search iterations") +
  ylab("Normalized test accuracy of best \n model (on valid set) up to this iteration") +
  theme_minimal(base_size=22) +
  theme(legend.position="none") +
  colScale

ggsave("analyses/plots/random_search_classif_numerical.jpg", width=7, height=6)


################################################################################


# Benchmark classification numerical LARGE

df <-  read_csv("results/sweeps/sweeps_classif/benchmark_classif_random_large.csv") %>% 
  mutate(hp = "random") %>% 
  bind_rows(read_csv("results/sweeps/sweeps_classif/benchmark_classif_default_large.csv") %>% 
              mutate(hp="default")) %>% 
  filter(data__keyword != "poker") %>% 
  #select(model_name, data__keyword, mean_val_score, mean_test_score, mean_time, hp) %>% 
  filter(!is.na(mean_test_score), !is.na(mean_val_score), !is.na(model_name)) %>% 
  rename()

df %>% 
  ungroup() %>% 
  mutate(n_iter = case_when(
    n_test > 6000 ~ 1,
    n_test > 3000 ~ 2,
    n_test > 1000 ~ 3,
    TRUE ~5
  )) %>% 
  summarise(s =  sum(mean_time * n_iter) / 3600)

#res_datasets <- 
#  df %>% 
#  filter(!is.na(mean_time)) %>% 
#  normalize_no_variable(normalization_type = "quantile", quantile = 0.3) %>% 
#  random_search_no_variable(n_shuffles=15, default_first = T)


res_datasets <- 
  df %>% 
  filter(!is.na(mean_time)) %>% 
  #normalize_no_variable(normalization_type = "quantile", quantile = 0.1) %>% 
  random_search_no_variable(n_shuffles=15, default_first = T) 


res_datasets %>% 
  group_by(random_rank, model_name, data__keyword) %>% 
  summarise(min_test_score = min(mean_test_score, na.rm=T),#quantile(mean_test_score, 0.1, na.rm=T),
            max_test_score = max(mean_test_score, na.rm=T),#quantile(mean_test_score, 0.9, na.rm=T),
            mean_test_score = mean(mean_test_score, na.rm=T)) %>% 
  ggplot() +
  #geom_hline(aes(yintercept=mean_test_score, color=model_name), data=res_datasets_ %>% filter(random_rank == 0), linetype="dotted", size=1.5, alpha=0.7) +
  geom_line(aes(x = random_rank, y = mean_test_score, color = model_name), size=2) +
  geom_ribbon(aes(x=random_rank, ymin = min_test_score, ymax=max_test_score, fill=model_name), alpha=0.3) +
  #geom_text(aes(x=150, y=mean_test_score + 0.0007, color=model_name, label=str_c(model_name, " (Default)")), data=res_ %>% filter(random_rank == 0), size=7) +
  #geom_dl(aes(label = model_name, x=random_rank, y=mean_test_score, color=model_name), method = list(dl.combine("smart.grid"), cex=1.3))  +
  facet_wrap(~data__keyword, scales="free") +
  #coord_cartesian(ylim=c(0.5, 1.0)) + 
  scale_x_log10(limits=c(1, 1000)) +
  xlab("Number of random search iterations") +
  ylab("Normalized test score of best model \n (on valid set) up to this iteration") +
  theme_minimal(base_size=22) +
  #theme(legend.position="none") +
  colScale
  

res_datasets_ <- res_datasets %>% 
  group_by(random_rank, model_name, n_dataset) %>% 
  summarise(mean_test_score = mean(mean_test_score, na.rm=T)) %>% #summarise on datasets
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
  #geom_dl(aes(label = model_name, x=random_rank, y=mean_test_score, color=model_name), method = list(dl.combine("smart.grid"), cex=1.3))  +
  geom_text_repel(aes(label=model_name, 
                      color = model_name,
                      x = random_rank,
                      y =  mean_test_score),
                  data= (res_datasets_ %>% 
                           filter(random_rank == 0)),
                  bg.color='white', size = 6,
                  nudge_y = 0.03, nudge_x = 0.2, min.segment.length=100)+
  coord_cartesian(ylim=c(0.6, 1.0)) + 
  scale_x_log10(limits=c(1, 260)) +
  xlab("Number of random search iterations") +
  ylab("Normalized test score of best model \n (on valid set) up to this iteration") +
  theme_minimal(base_size=22) +
  theme(legend.position="none") +
  colScale

ggsave("analyses/plots/random_search_classif_numerical_large.jpg", width=7, height=6)


View(df %>% group_by(model_name, data__keyword) %>% summarise(count = n()))


###################################
# Benchmark classif categorical

# Medium

df <-  read_csv("results/sweeps/sweeps_classif/benchmark/ft_transformer_categorical_classif.csv") %>% 
  bind_rows(read_csv("results/sweeps/sweeps_classif/benchmark/resnet_categorical_classif.csv")) %>% 
  bind_rows(read_csv("results/sweeps/sweeps_classif/benchmark/rf_categorical_classif.csv")) %>% 
  bind_rows(read_csv("results/sweeps/sweeps_classif/benchmark/gbt_categorical_classif.csv")) %>% 
  bind_rows(read_csv("results/sweeps/sweeps_classif/benchmark/hgbt_categorical_classif.csv")) %>% 
  bind_rows(read_csv("results/sweeps/sweeps_classif/xgb_categorical_classif.csv")) %>% 
  bind_rows(read_csv("results/sweeps/sweeps_classif/saint_categorical_classif.csv") %>% 
              mutate(model_name = "SAINT")) %>% 
  mutate(hp = "random") %>% 
  bind_rows(read_csv("results/sweeps/sweeps_classif/benchmark_categorical_classif_medium_default.csv") %>% 
              bind_rows(read_csv("results/sweeps/sweeps_classif/benchmark/xgb_categorical_classif_default.csv")) %>% 
              bind_rows(read_csv("results/sweeps/sweeps_classif/benchmark/saint_categorical_classif_default.csv") %>% 
                          mutate(model_name = "SAINT")) %>% 
              mutate(hp = "default")) %>% 
  #bind_rows(read_csv("results/sweeps/sweeps_classif/benchmark_classif_default_large.csv") %>% 
  #            mutate(hp="default")) %>% 
  #filter(data__keyword != "poker") %>% 
  #select(model_name, data__keyword, mean_val_score, mean_test_score, mean_time, hp) %>% 
  filter(!is.na(mean_test_score), !is.na(mean_val_score), !is.na(model_name)) %>% 
  rename()

# Large
df <- read_csv("results/sweeps/sweeps_classif/benchmark/large/gbt_categorical_classif_large.csv") %>% 
  bind_rows(read_csv("results/sweeps/sweeps_classif/benchmark/large/hgbt_categorical_classif_large.csv")) %>% 
  bind_rows(read_csv("results/sweeps/sweeps_classif/benchmark/large/rf_categorical_classif_large.csv")) %>% 
  bind_rows(read_csv("results/sweeps/sweeps_classif/benchmark/large/xgb_categorical_classif_large.csv")) %>% 
  bind_rows(read_csv("results/sweeps/sweeps_classif/benchmark/large/saint_categorical_classif_large.csv")) %>% 
  bind_rows(read_csv("results/sweeps/sweeps_classif/benchmark/large/resnet_categorical_classif_large.csv")) %>% 
  bind_rows(read_csv("results/sweeps/sweeps_classif/benchmark/large/ft_categorical_classif_large.csv")) %>% 
  mutate(hp = "random") %>% 
  select(model_name, data__keyword, mean_test_score, mean_val_score, mean_time, hp) %>% 
  filter(!is.na(mean_test_score), !is.na(mean_val_score), !is.na(model_name)) %>% 
  rename()



  
# Measure time spent
# 
# df %>% 
#   ungroup() %>% 
#   mutate(n_iter = case_when(
#     n_test > 6000 ~ 1,
#     n_test > 3000 ~ 2,
#     n_test > 1000 ~ 3,
#     TRUE ~5
#   )) %>% 
#   summarise(s =  sum(mean_time * n_iter) / 3600)

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
  group_by(random_rank, model_name, data__keyword) %>% 
  summarise(min_test_score = min(mean_test_score, na.rm=T),#quantile(mean_test_score, 0.1, na.rm=T),
            max_test_score = max(mean_test_score, na.rm=T),#quantile(mean_test_score, 0.9, na.rm=T),
            mean_test_score = mean(mean_test_score, na.rm=T)) %>% 
  ggplot() +
  #geom_hline(aes(yintercept=mean_test_score, color=model_name), data=res_datasets_ %>% filter(random_rank == 0), linetype="dotted", size=1.5, alpha=0.7) +
  geom_line(aes(x = random_rank, y = mean_test_score, color = model_name), size=2) +
  geom_ribbon(aes(x=random_rank, ymin = min_test_score, ymax=max_test_score, fill=model_name), alpha=0.3) +
  #geom_text(aes(x=150, y=mean_test_score + 0.0007, color=model_name, label=str_c(model_name, " (Default)")), data=res_ %>% filter(random_rank == 0), size=7) +
  #geom_dl(aes(label = model_name, x=random_rank, y=mean_test_score, color=model_name), method = list(dl.combine("smart.grid"), cex=1.3))  +
  facet_wrap(~data__keyword, scales="free") +
  #coord_cartesian(ylim=c(0.5, 1.0)) + 
  scale_x_log10(limits=c(1, 1000)) +
  xlab("Number of random search iterations") +
  ylab("Normalized test score of best model \n (on valid set) up to this iteration") +
  theme_minimal(base_size=22) +
  #theme(legend.position="none") +
  colScale


res_datasets_ <- res_datasets %>% 
  #filter(data__keyword != "LoanDefaultPrediction", 
  #       data__keyword !)
  group_by(random_rank, model_name, n_dataset) %>% 
  summarise(mean_test_score = mean(mean_test_score, na.rm=T)) %>% #summarise on datasets
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
  #geom_dl(aes(label = model_name, x=random_rank, y=mean_test_score, color=model_name), method = list(dl.combine("smart.grid"), cex=1.3))  +
  geom_text_repel(aes(label=model_name, 
                      color = model_name,
                      x = random_rank,
                      y =  mean_test_score),
                  data= (res_datasets_ %>% 
                           filter(random_rank == 1)),
                  bg.color='white', size = 6.5, bg.r=0.15,
                  nudge_y = 0., nudge_x = 0.3, min.segment.length=100)+
  coord_cartesian(ylim=c(0.4, 1.0)) + 
  scale_x_log10(limits=c(1, 700)) +
  xlab("Number of random search iterations") +
  ylab("Normalized test accuracy of best  \n model (on valid set) up to this iteration") +
  theme_minimal(base_size=22) +
  theme(legend.position="none") +
  colScale

ggsave("analyses/plots/benchmark_categorical_classif.jpg", width=7, height=6)



