library(tidyverse)

total_df <-read_csv("results/new/sparse_new_real_01_02_01.csv") %>% 
  bind_rows(read_csv("results/new/sparse_new_real_01_02_02.csv")) %>% 
  bind_rows(read_csv("results/new/sparse_new_real_01_02_03.csv")) 
  

total_df %>% 
  filter(is.na(module__dropout_prob),
         module__n_layers==2,
         module__bias==T,
         #module__concatenate_input==F,
         module__temperature == 0.5) %>% 
  mutate(module__concatenate_input=as_factor(module__concatenate_input)) %>% 
  #filter(method_name == "sparse") %>% 
  #filter(method_name %in% c("mlp", "lounici", "nam", "auto_sklearn", "rf", "hgbt", "resnet")) %>% 
  #filter(method_name %in% c("saint")) %>% 
  #mutate(scheduler = if_else(is.na(scheduler), F, T)) %>% 
  filter(train_scores > 0.8) %>% 
  pivot_longer(cols = c(test_scores, train_scores), values_to="score", names_to = "score_type") %>% 
  filter(transform__1__method_name == "no_transform", transform__0__method_name == "gaussienize", transform__0__type == "quantile") %>%
  #group_by(method_name, data__keyword, score_type, model_params_str, module__temperature) %>% 
  #summarise(mean_score_iter = mean(score, na.rm=T)) %>% 
  #ungroup() %>%
  #filter(module__n_w == 5 | is.na(module__n_w),
  #       module__activation_on == "each" | is.na(module__activation_on)) %>% 
  #group_by(data__keyword, score_type) %>% 
  #mutate(relative_score = max_score_hp / max(max_score_hp)) %>% 
  ggplot() +
  #geom_point(aes(y = data__keyword, x = mean_score_iter, color = module__n_layers, shape = module__n_w, group=module__n_w), alpha=0.5, position = position_dodge(width=1)) +
  geom_point(aes(y = score, x = module__concatenate_input, color = data__keyword), alpha=0.6) +
  #geom_jitter(aes(y = data__keyword, x = score, color = algo_hp), alpha=0.5, width=0, height=0.1)+
  facet_wrap(~score_type) +
  xlab("Concatenate input") +
  ylab("Score")

