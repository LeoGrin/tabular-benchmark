source("analyses/random_search_utils.R")

read_csv("analyses/dataset numerical variables - Feuille 8.csv") %>% 
  left_join(read_csv("analyses/dataset numerical variables - categorical_classif.csv"), by = c("dataset_id"), suffix=c("",".y"))%>%
  select(-ends_with(".y")) %>% 
  write_csv("dataset_classif_categorical.csv")

df <- read_csv("results/sweeps/sweeps_classif/benchmark/medium/gbt_classif.csv") %>%
  bind_rows(read_csv("results/sweeps/sweeps_classif/benchmark/medium/rf_classif.csv")) %>%  
  bind_rows(read_csv("results/sweeps/sweeps_classif/benchmark/medium/xgb_classif.csv")) %>%
  bind_rows(read_csv("results/sweeps/sweeps_classif/benchmark_classif_random_medium_checkpoint.csv")) %>% 
              #select(model_name, data__keyword, mean_val_score, mean_test_score, mean_time)) %>% 
  bind_rows(read_csv("results/sweeps/sweeps_classif/benchmark/medium/xgb_random_jannis.csv")) %>%
  bind_rows(read_csv("results/sweeps/sweeps_classif/benchmark/benchmark_classif_random_medium_numerical_bonus.csv")) %>% 
  bind_rows(read_csv("results/sweeps/sweeps_classif/benchmark/benchmark_classif_random_medium_bonus_2.csv")) %>% 
  mutate(hp="random") %>% 
  #bind_rows(
  #  read_csv("results/sweeps/sweeps_classif/benchmark/medium/gbt_classif_default.csv") %>%
  #    bind_rows(read_csv("results/sweeps/sweeps_classif/benchmark/medium/rf_classif_default.csv")) %>% 
  #    bind_rows(read_csv("results/sweeps/sweeps_classif/benchmark/benchmark_classif_default_medium_numerical.csv")) %>% 
  #    mutate(hp="default")) %>% 
  filter(data__keyword != "poker") %>% 
  #select(model_name, data__keyword, mean_val_score, mean_test_score, mean_time, hp) %>% 
  filter(!is.na(mean_test_score), !is.na(mean_val_score), !is.na(model_name)) %>% 
  rename()

#res_datasets <- 
#  df %>% 
#  filter(!is.na(mean_time)) %>% 
#  normalize_no_variable(normalization_type = "quantile", quantile = 0.3) %>% 
#  random_search_no_variable(n_shuffles=15, default_first = T)

quantile <- 0.1

model <- "Resnet"

View(df %>% 
       #filter(!is.na(mean_time), !is.na(mean_test_score)) %>% 
       #filter(model_name == model) %>% 
  mutate(model__module__d_embedding = cut_number(model__module__d_embedding, 5)) %>% 
  group_by(model__module__d_embedding, data__keyword) %>% 
  summarise(sd_test_score = sd(mean_test_score), mean_test_score = mean(mean_test_score)))

quantile <- 0.1
View(df %>% 
  filter(!is.na(mean_test_score)) %>% 
  filter(model_name == model) %>% 
  select_if(~sum(!is.na(.)) > 0) %>% 
  select(data__keyword, mean_val_score, mean_test_score, starts_with("model__")) %>% 
  select_if(~length(unique(.)) > 1) %>% 
  group_by(data__keyword) %>% 
  mutate(q = quantile(mean_test_score, quantile, na.rm=T), m =max(mean_test_score, na.rm=T)) %>% 
  mutate(mean_test_score_ = (mean_test_score - quantile(mean_test_score, quantile, na.rm=T)) / (max(mean_test_score, na.rm=T) - quantile(mean_test_score, quantile, na.rm=T))) %>% 
  mutate(mean_test_score2 = (mean_test_score - q) / (m - q)) %>% 
  ungroup() %>% 
  #mutate(model__module__d_embedding = cut_number(model__module__d_embedding, 5))%>% 
  select(model__module__d_embedding, data__keyword, mean_test_score2, mean_test_score_, mean_test_score, q, m))
  #group_by(model__module__d_embedding, data__keyword) %>% 
  #summarise(mean_test_score_ = mean(mean_test_score_), mean_test_score = mean(mean_test_score), mean_q = mean(q), mean_m = mean(m)))

df_ <- df %>% 
  filter(!is.na(mean_time), !is.na(mean_test_score)) %>% 
  select_if(~sum(!is.na(.)) > 0) %>% 
  select(data__keyword, mean_val_score, mean_test_score, starts_with("model__")) %>% 
  select_if(~length(unique(.)) > 1) %>% 
  mutate(model_name = model) %>% 
  filter(data__keyword == "credit") %>% 
 # group_by(data__keyword) %>% 
  mutate(mean_val_score = (mean_val_score - quantile(mean_val_score, quantile, na.rm=T)) / (max(mean_val_score, na.rm=T) - quantile(mean_val_score, quantile, na.rm=T)), 
       mean_test_score = (mean_test_score - quantile(mean_test_score, quantile, na.rm=T)) / (max(mean_test_score, na.rm=T) - quantile(mean_test_score, quantile, na.rm=T))) %>% 
  ungroup() %>% 
  mutate(model__module__d_embedding = cut_number(model__module__d_embedding, 5)) %>% 
  random_search(variable = model__module__d_embedding, n_shuffles=15, default_first = F) 
  
#vars <- #c("model__lr", "model__module__normalization",
        #  "model__module__n_layers")

vars <- c("model__lr", "model__module__d_embedding")

df <- df %>% 
  filter(!is.na(mean_time), !is.na(mean_test_score)) %>% 
  #mutate(model_name = model) %>% 
  group_by(data__keyword) %>% 
  mutate(mean_val_score = (mean_val_score - quantile(mean_val_score, quantile, na.rm=T)) / (max(mean_val_score, na.rm=T) - quantile(mean_val_score, quantile, na.rm=T)), 
          mean_test_score = (mean_test_score - quantile(mean_test_score, quantile, na.rm=T)) / (max(mean_test_score, na.rm=T) - quantile(mean_test_score, quantile, na.rm=T))) %>% 
  ungroup() %>% 
  filter(model_name == model) %>% 
  select_if(~sum(!is.na(.)) > 0) %>% 
  select(data__keyword, mean_val_score, mean_test_score, starts_with("model__")) %>% 
  select_if(~length(unique(.)) > 1) %>% 
  mutate(model_name = model)

res <- tibble()
for (var in vars){
  print(var)
  print(typeof(var))
  if (startsWith(var, "model__")){
    df_ <-  df %>% 
      mutate(variable = !!as.symbol(var))
    if (n_distinct(df_$variable) < 5) {
      res_dataset <- df_ %>% 
        mutate(variable = as_factor(variable))
    }
    else {
      res_dataset <- df_ %>% 
        mutate(variable = cut_number(variable, 5))
    }
    res_dataset <- 
      res_dataset %>% 
      random_search(variable = variable, n_shuffles=15, default_first = F) %>% 
      mutate(variable_fixed = var)
    #View(res_dataset %>% select(mean_val_score, mean_test_score, best_val_score_so_far, variable, data__keyword, random_rank, n_dataset, variable_fixed))

    res <- res %>% bind_rows(res_dataset)
  }
}

res %>% 
  filter(random_rank == 60) %>% 
  group_by(model_name, variable,  variable_fixed, n_dataset) %>% 
  summarise(mean_test_score = mean(mean_test_score)) %>% 
  ggplot() +
  geom_boxplot(aes(x = variable, y = mean_test_score)) +
  facet_wrap(~variable_fixed, scales = "free")

res %>% 
  filter(random_rank == 50) %>% 
  #group_by(model_name, variable,  variable_fixed, n_dataset) %>% 
  #summarise(mean_test_score = mean(mean_test_score)) %>% 
  ggplot() +
  geom_boxplot(aes(x = variable, y = mean_test_score)) +
  facet_wrap(data__keyword~variable_fixed, scales = "free")
  
  

res_datasets <- df %>% 
filter(!is.na(mean_time), !is.na(mean_test_score)) %>% 
group_by(data__keyword) %>% 
mutate(mean_val_score = (mean_val_score - quantile(mean_val_score, quantile, na.rm=T)) / (max(mean_val_score, na.rm=T) - quantile(mean_val_score, quantile, na.rm=T)), 
       mean_test_score = (mean_test_score - quantile(mean_test_score, quantile, na.rm=T)) / (max(mean_test_score, na.rm=T) - quantile(mean_test_score, quantile, na.rm=T))) %>% 
ungroup() %>% 
filter(model_name == "Resnet") %>%
mutate(model__lr = cut_number(model__module__d_hidden_factor, 5)) %>% 
random_search(variable=model__lr, n_shuffles = 15, default_first = F)


res_datasets %>% 
  filter(data__keyword != "poker") %>% #, data__keyword %in% c("electricity", "california")) %>% 
  group_by(model_name, random_rank, model__lr, n_dataset) %>% 
  summarise(mean_test_score = mean(mean_test_score, na.rm=T)) %>% 
  summarise(max_test_score = max(mean_test_score, na.rm=T),
            min_test_score = min(mean_test_score, na.rm=T),
            mean_test_score = mean(mean_test_score, na.rm=T)) %>% 
  mutate(model__lr = as_factor(model__lr)) %>% 
  ungroup() %>% 
  ggplot() +
  geom_line(aes(x = random_rank, y=mean_test_score, color=model__lr), size=2) + #, width=0.1, height=0) +
  geom_ribbon(aes(x = random_rank, y = mean_test_score, ymax=max_test_score, ymin = min_test_score, fill=model__lr), alpha=0.3) +
  scale_x_log10()+
  coord_cartesian(ylim=c(0.3, 1.0)) + 
  theme_minimal(base_size=22)
  
  #colScale
#facet_wrap(~data__keyword, scales = "free")


res_dataset <- df %>% 
  filter(!is.na(mean_time), !is.na(mean_test_score)) %>% 
  group_by(data__keyword) %>% 
  mutate(mean_val_score = (mean_val_score - quantile(mean_val_score, quantile, na.rm=T)) / (max(mean_val_score, na.rm=T) - quantile(mean_val_score, quantile, na.rm=T)), 
         mean_test_score = (mean_test_score - quantile(mean_test_score, quantile, na.rm=T)) / (max(mean_test_score, na.rm=T) - quantile(mean_test_score, quantile, na.rm=T))) %>% 
  filter(model_name == "MLP")
  

dataset <- "electricity"

cor.test(x=(res_dataset %>% 
              filter(data__keyword == dataset))$mean_test_score, y=(res_dataset %>% 
                                                                      filter(data__keyword == dataset))$model__batch_size, method = 'spearman')
