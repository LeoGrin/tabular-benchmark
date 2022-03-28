library(tidyverse)

total_df <- read_csv("results/new/res_new_real_bonus_2_10_11.csv") %>% 
  filter(method_name %in% c("auto_sklearn", "rf", "hgbt")) %>% 
  #bind_rows(read_csv("results/new/mlp_03_01.csv")) %>% 
  bind_rows(read_csv("results/new/sparse_new_real_01_02_01.csv")) %>% 
  bind_rows(read_csv("results/new/sparse_new_real_01_02_02.csv")) %>% 
  bind_rows(read_csv("results/new/sparse_new_real_01_02_03.csv")) %>% 
  bind_rows(read_csv("results/new/sparse_new_real_no_train_01_02.csv") %>% 
              mutate(method_name = "sparse_new_no_train"))
  #mutate(method_name = if_else(method_name == "mlp", "mlp_old", method_name)) %>% 
  #bind_rows(read_csv("results/new/sparse_new_real_26_01_4.csv") %>% 
  #            filter(module__train_selectors == T) %>% 
              #filter(module__temperature == 0.5) %>% 
  #            filter(module__linear_output_layer == F))


  #bind_rows(read_csv("results/new/mlp_electricity_new.csv"))
  #bind_rows(read_csv("results/new/mlp_real_data_05_01_no_resnet.csv") %>% mutate(method_name = "mlp_no_resnet")) %>%
  #bind_rows(read_csv("results/new/mlp_real_data_05_01_resnet.csv") %>% mutate(method_name = "mlp_resnet"))

#View(read_csv("results/new/res_new_real_bonus_2_10_11.csv"))
read_csv("results/new/mlp_03_01.csv") %>% select(test_scores, train_scores, data__keyword, module__dropout_prob)




total_df %>% 
  filter(data__keyword != "electricity_new") %>% 
  #filter(method_name == "sparse") %>% 
  #filter(method_name %in% c("mlp", "lounici", "nam", "auto_sklearn", "rf", "hgbt", "resnet")) %>% 
  #filter(method_name %in% c("saint")) %>% 
  #mutate(scheduler = if_else(is.na(scheduler), F, T)) %>% 
  filter(train_scores > 0.8) %>% 
  pivot_longer(cols = c(test_scores, train_scores), values_to="score", names_to = "score_type") %>% 
  filter(transform__1__method_name == "no_transform", transform__0__method_name == "gaussienize", transform__0__type == "quantile") %>%
  group_by(method_name, data__keyword, score_type, model_params_str, module__temperature) %>% 
  summarise(mean_score_iter = mean(score, na.rm=T)) %>% 
  ungroup() %>%
  #filter(module__n_w == 5 | is.na(module__n_w),
  #       module__activation_on == "each" | is.na(module__activation_on)) %>% 
  #group_by(data__keyword, score_type) %>% 
  #mutate(relative_score = max_score_hp / max(max_score_hp)) %>% 
  ggplot() +
  #geom_point(aes(y = data__keyword, x = mean_score_iter, color = module__n_layers, shape = module__n_w, group=module__n_w), alpha=0.5, position = position_dodge(width=1)) +
  geom_jitter(aes(y = data__keyword, x = mean_score_iter, color = method_name, shape=method_name, group=method_name), alpha=0.8, position = position_dodge(width=1)) +
  #geom_jitter(aes(y = data__keyword, x = score, color = algo_hp), alpha=0.5, width=0, height=0.1)+
  facet_wrap(~score_type) +
  xlab("Mean score on > 10 iter") +
  ylab("Dataset")

  
total_df %>% 
  filter(method_name %in% c("mlp", "lounici", "nam", "auto_sklearn", "rf", "hgbt")) %>% 
  filter(data__keyword != "heloc") %>% 
  #mutate(scheduler = if_else(is.na(scheduler), F, T)) %>% 
  pivot_longer(cols = c(test_scores, train_scores), values_to="score", names_to = "score_type") %>% 
  filter(score_type == "test_scores") %>% 
  filter(transform__1__method_name == "no_transform", transform__0__method_name == "gaussienize", transform__0__type == "quantile") %>%
  group_by(method_name, data__keyword, score_type, model_params_str) %>% 
  summarise(mean_score_iter = mean(score, na.rm=T)) %>% 
  ungroup() %>% 
  group_by(method_name, data__keyword, score_type) %>% 
  summarise(best_score = max(mean_score_iter, na.rm=T)) %>% 
  ungroup() %>% 
  group_by(data__keyword, score_type) %>% 
  mutate(relative_score = best_score / max(best_score, na.rm=T)) %>% 
  mutate(rank = rank(-best_score)) %>% 
  ungroup() %>% 
  group_by(method_name) %>% 
  summarise(mean_rank = mean(rank), sd_rank = sd(rank), mean_rel_score = mean(relative_score), sd_rel_score = sd(relative_score)) %>% 
  arrange(mean_rank)


#saint

saint <- read_csv("results/new/saint.csv") %>% 
  mutate(method_name = "saint")

saint$dataset <- c("heloc", "electricity", "california", "covtype", "spam", "churn", "credit", "shopping", "nomao", "cpu", "wine")

saint <- saint %>% bind_rows(read_csv("results/new/saint_2.csv") %>% 
  mutate(method_name = "saint_2")) %>% 
  bind_rows(read_csv("results/new/saint_3.csv") %>% mutate(method_name = "saint_3"))

total_df %>% 
  filter(method_name %in% c("mlp", "auto_sklearn", "rf", "hgbt")) %>% 
  #filter(method_name %in% c("saint")) %>% 
  #mutate(scheduler = if_else(is.na(scheduler), F, T)) %>% 
  pivot_longer(cols = c(test_scores, train_scores), values_to="score", names_to = "score_type") %>% 
  filter(transform__1__method_name == "no_transform", transform__0__method_name == "gaussienize", transform__0__type == "quantile") %>%
  group_by(method_name, data__keyword, score_type, model_params_str, scheduler) %>% 
  summarise(mean_score_iter = mean(score, na.rm=T)) %>% 
  ungroup() %>% 
  bind_rows(saint %>% group_by(method_name, dataset) %>% summarise(mean_score_iter = mean(best_accuracy, na.rm=T) / 100, 
                             data__keyword=dataset, 
                             score_type = "test_scores")) %>% 
  #group_by(data__keyword, score_type) %>% 
  #mutate(relative_score = max_score_hp / max(max_score_hp)) %>% 
  mutate(is_saint = method_name == "saint" | method_name == "saint_2" | method_name == "saint_3") %>% 
  ggplot() +
  geom_point(aes(y = data__keyword, x = mean_score_iter, color = method_name, shape=is_saint), alpha=1, position = position_dodge(width=0.5)) +
  #geom_jitter(aes(y = data__keyword, x = score, color = algo_hp), alpha=0.5, width=0, height=0.1)+
  facet_wrap(~score_type) +
  xlab("Mean score on > 10 iter") +
  ylab("Dataset")

saint_datasets <- c("covtype", "spam", "churn", "shopping")

other_datasets <- c("heloc", "electricity", "california", "credit", "nomao", "cpu", "wine")


total_df %>% 
  filter(method_name %in% c("mlp", "auto_sklearn", "rf", "hgbt")) %>% 
  #filter(method_name %in% c("saint")) %>% 
  #mutate(scheduler = if_else(is.na(scheduler), F, T)) %>% 
  pivot_longer(cols = c(test_scores, train_scores), values_to="score", names_to = "score_type") %>% 
  filter(transform__1__method_name == "no_transform", transform__0__method_name == "gaussienize", transform__0__type == "quantile") %>%
  group_by(method_name, data__keyword, score_type, model_params_str, scheduler) %>% 
  summarise(mean_score_iter = mean(score, na.rm=T)) %>% 
  ungroup() %>% 
  bind_rows(saint %>% mutate(mean_score_iter = best_accuracy / 100, 
                             data__keyword=dataset, 
                             score_type = "test_scores")) %>% 
  group_by(data__keyword, score_type, method_name) %>% 
  summarise(mean_score_iter = max(mean_score_iter)) %>% #max on hp
  group_by(data__keyword, score_type) %>% 
  mutate(relative_score = mean_score_iter / max(mean_score_iter, na.rm=T)) %>% 
  ungroup() %>% 
  filter(score_type == "test_scores") %>% 
  group_by(method_name) %>% 
  summarise(mean_rel_score = mean(relative_score, na.rm=T), sd_rel_score = sd(relative_score, na.rm=T)) %>% 
  arrange(mean_rel_score)

total_df %>% 
  filter(method_name %in% c("mlp", "auto_sklearn", "rf", "hgbt")) %>% 
  #filter(method_name %in% c("saint")) %>% 
  #mutate(scheduler = if_else(is.na(scheduler), F, T)) %>% 
  pivot_longer(cols = c(test_scores, train_scores), values_to="score", names_to = "score_type") %>% 
  filter(transform__1__method_name == "no_transform", transform__0__method_name == "gaussienize", transform__0__type == "quantile") %>%
  group_by(method_name, data__keyword, score_type, model_params_str, scheduler) %>% 
  summarise(mean_score_iter = mean(score, na.rm=T)) %>% 
  ungroup() %>% 
  bind_rows(saint %>% mutate(mean_score_iter = best_accuracy / 100, 
                             data__keyword=dataset, 
                             score_type = "test_scores")) %>% 
  group_by(data__keyword, score_type, method_name) %>% 
  summarise(mean_score_iter = max(mean_score_iter)) %>% #max on hp
  group_by(data__keyword, score_type) %>% 
  mutate(relative_score = mean_score_iter / max(mean_score_iter, na.rm=T)) %>% 
  ungroup() %>% 
  filter(score_type == "test_scores", method_name %in% c("saint", "saint_2")) %>%
  mutate(dataset_in_paper = data__keyword %in% saint_datasets) %>% 
  group_by(dataset_in_paper, method_name) %>% 
  summarise(mean_rel_score = mean(relative_score, na.rm=T), sd_rel_score = sd(relative_score, na.rm=T)) %>% 
  arrange(method_name, mean_rel_score)
  


total_df %>% 
  filter(method_name %in% c("mlp", "auto_sklearn", "rf", "hgbt")) %>% 
  #filter(method_name %in% c("saint")) %>% 
  #mutate(scheduler = if_else(is.na(scheduler), F, T)) %>% 
  pivot_longer(cols = c(test_scores, train_scores), values_to="score", names_to = "score_type") %>% 
  filter(transform__1__method_name == "no_transform", transform__0__method_name == "gaussienize", transform__0__type == "quantile") %>%
  group_by(method_name, data__keyword, score_type, model_params_str, scheduler) %>% 
  summarise(mean_score_iter = mean(score, na.rm=T)) %>% 
  ungroup() %>% 
  bind_rows(saint %>% mutate(mean_score_iter = best_accuracy / 100, 
                             data__keyword=dataset, 
                             score_type = "test_scores")) %>% 
  group_by(data__keyword, score_type, method_name) %>% 
  summarise(mean_score_iter = max(mean_score_iter)) %>% #max on hp
  group_by(data__keyword, score_type) %>% 
  mutate(relative_score = mean_score_iter / max(mean_score_iter, na.rm=T)) %>% 
  ungroup() %>% 
  mutate(dataset_in_paper = data__keyword %in% saint_datasets) %>% 
  filter(score_type == "test_scores", dataset_in_paper==F) %>%
  group_by(method_name) %>% 
  summarise(mean_rel_score = mean(relative_score, na.rm=T), sd_rel_score = sd(relative_score, na.rm=T)) %>% 
  arrange(mean_rel_score)

