library(tidyverse)

total_df <- read_csv("results/new/res_new_real_bonus_2_10_11.csv") %>% 
  filter(method_name %in% c("mlp","auto_sklearn", "rf", "hgbt")) %>% 
  filter(data__keyword != "electricity_new") %>% 
  mutate(method_name = if_else(method_name == "mlp", "mlp_old", method_name)) %>% 
  #bind_rows(read_csv("results/new/lounici_05_12.csv")) %>% 
  #bind_rows(read_csv("results/new/lounici_scheduler_05_12.csv")) %>% 
  #bind_rows(read_csv("results/new/nam.csv")) %>% 
  #bind_rows(read_csv("results/new/nam_embedding.csv")) %>% 
  #bind_rows(read_csv("results/new/mlp_05_12.csv")) %>% 
  #bind_rows(read_csv("results/new/resnet.csv") %>% mutate(method_name="resnet")) %>% 
  #bind_rows(read_csv("results/sparse.csv")) %>% 
  bind_rows(read_csv("results/new/mlp_03_01.csv")) %>% 
  bind_rows(read_csv("results/new/default_new_heloc_elec.csv")) %>% 
  #bind_rows(read_csv("results/new/mlp_electricity_new.csv")) %>% 
  #bind_rows(read_csv("results/new/sparse_real_10_01.csv")) %>% 
  bind_rows(read_csv("results/new/sparse_real_17_01.csv") %>% 
              mutate(method_name = "sparse_new")) %>% 
  bind_rows(read_csv("results/new/sparse_real_17_01_bonus_2.csv") %>% 
              mutate(method_name = "sparse_new"))
#bind_rows(read_csv("results/new/sparse_real_17_01_3.csv") %>% 
#            mutate(method_name = "sparse_new")) %>% 
#bind_rows(read_csv("results/new/sparse_real_17_01_4.csv") %>% 
#            mutate(method_name = "sparse_new")) 
#bind_rows(read_csv("results/new/mlp_real_data_05_01_no_resnet.csv") %>% mutate(method_name = "mlp_no_resnet")) %>%
#bind_rows(read_csv("results/new/mlp_real_data_05_01_resnet.csv") %>% mutate(method_name = "mlp_resnet"))

#View(read_csv("results/new/res_new_real_bonus_2_10_11.csv"))
View(read_csv("results/new/mlp_03_01.csv") %>% select(test_scores, train_scores, data__keyword, module__dropout_prob))




total_df %>% 
  #filter(method_name %in% c("mlp", "lounici", "nam", "auto_sklearn", "rf", "hgbt", "resnet")) %>% 
  
  #filter(method_name %in% c("saint")) %>% 
  #mutate(scheduler = if_else(is.na(scheduler), F, T)) %>% 
  pivot_longer(cols = c(test_scores, train_scores), values_to="score", names_to = "score_type") %>% 
  filter(transform__1__method_name == "no_transform", transform__0__method_name == "gaussienize", transform__0__type == "quantile") %>%
  group_by(method_name, data__keyword, score_type, model_params_str, module__dropout_prob) %>% 
  summarise(mean_score_iter = mean(score, na.rm=T)) %>% 
  ungroup() %>% 
  mutate(module__dropout_prob = if_else(is.na(module__dropout_prob), 0, module__dropout_prob)) %>%
  mutate(module__dropout_prob = as_factor(module__dropout_prob)) %>% 
  mutate(is_sparse = if_else(method_name == "sparse", T, F)) %>% 
   #group_by(data__keyword, score_type) %>% 
  #mutate(relative_score = max_score_hp / max(max_score_hp)) %>% 
  ggplot() +
  geom_point(aes(y = data__keyword, x = mean_score_iter, color = method_name, shape = module__temperature, group=method_name), alpha=1, position = position_dodge(width=1)) +
  #geom_jitter(aes(y = data__keyword, x = score, color = algo_hp), alpha=0.5, width=0, height=0.1)+
  facet_wrap(~score_type) +
  xlab("Mean score on > 10 iter") +
  ylab("Dataset")



library(tidyverse)

total_df <- read_csv("results/new/res_new_real_bonus_2_10_11.csv") %>% 
  filter(method_name %in% c("mlp","auto_sklearn", "rf", "hgbt")) %>% 
  bind_rows(read_csv("results/new/sparse_real_23_01.csv"))




total_df %>% 
  filter(data__keyword %in% c("electricity", "california")) %>% 
  #filter(module__temperature == "1.0" | is.na(module__temperature)) %>% 
  pivot_longer(cols = c(test_scores, train_scores), values_to="score", names_to = "score_type") %>% 
  filter(transform__1__method_name == "no_transform", transform__0__method_name == "gaussienize", transform__0__type == "quantile") %>%
  group_by(method_name, data__keyword, score_type, model_params_str, module__temperature) %>% 
  summarise(mean_score_iter = mean(score, na.rm=T)) %>% 
  ungroup() %>% 
  mutate(is_sparse = if_else(method_name == "sparse", T, F)) %>% 
  #group_by(data__keyword, score_type) %>% 
  #mutate(relative_score = max_score_hp / max(max_score_hp)) %>% 
  ggplot() +
  geom_point(aes(y = data__keyword, x = mean_score_iter, color = module__temperature, shape = is_sparse, group=method_name), alpha=0.3, position = position_dodge(width=0.2)) +
  #geom_jitter(aes(y = data__keyword, x = score, color = algo_hp), alpha=0.5, width=0, height=0.1)+
  facet_wrap(~score_type) +
  xlab("Mean score on > 10 iter") +
  ylab("Dataset")



View(total_df %>% 
  filter(data__keyword %in% c("california")) %>% 
  filter(model_name == "sparse") %>% 
  pivot_longer(cols = c(test_scores, train_scores), values_to="score", names_to = "score_type") %>% 
  filter(transform__1__method_name == "no_transform", transform__0__method_name == "gaussienize", transform__0__type == "quantile") %>%
  group_by(method_name, data__keyword, score_type, model_params_str) %>% 
  summarise(mean_score_iter = mean(score, na.rm=T)) %>% 
  ungroup() %>% 
  filter(score_type == "test_scores") %>% 
  arrange(-mean_score_iter))
  

