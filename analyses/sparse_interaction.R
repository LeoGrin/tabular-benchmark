library(tidyverse)

df <- read_csv("results/new/sparse_interaction_cpu.csv")

df %>% 
  pivot_longer(cols = c(test_scores, train_scores), values_to="score", names_to = "score_type") %>% 
  filter(data__cov_matrix=="identity",
         data__num_features==30,
         data__num_samples==5000,
         target__n_interactions==5,
         target__variant=="transform_sum",
         target__ensemble_size < 1.5) %>% 
  ggplot() +
  geom_point(aes(y = method_name, x = score, color = target__ensemble_size), alpha=1, position=position_dodge2(width=0.5, reverse=T)) +
  #geom_jitter(aes(y = data__keyword, x = score, color = algo_hp), alpha=0.5, width=0, height=0.1)+
  facet_wrap(~score_type) +
  xlab("Mean score on > 10 iter") +
  ylab("Dataset")



df <- read_csv("results/new/sparse_in_interaction_gpu_06_01_128.csv") %>% 
  bind_rows(read_csv("results/new/sparse_interaction_cpu_07_01.csv"))

df %>% 
  pivot_longer(cols = c(test_scores, train_scores), values_to="score", names_to = "score_type") %>% 
  filter(data__cov_matrix=="identity",
         data__num_features==30,
         data__num_samples==5000,
        # target__n_interactions==3,
         target__variant=="transform_sum",
         target__ensemble_size < 1.5) %>% 
  mutate(module__dropout_prob = if_else(is.na(module__dropout_prob), 0, module__dropout_prob)) %>% 
  mutate(module__dropout_prob = as_factor(module__dropout_prob)) %>% 
  group_by(method_name, target__ensemble_size, score_type, module__dropout_prob, target__n_interactions) %>% 
  mutate(mean_score = mean(score)) %>% 
  filter(score_type == "train_scores") %>% 
  ggplot() +
  geom_jitter(aes(x = target__ensemble_size, y = score, color = method_name), alpha=0.5, size=1)+
  geom_line(aes(x = target__ensemble_size, y = mean_score, color = method_name, linetype=module__dropout_prob))+
  facet_wrap(~target__n_interactions) +
  ggtitle("Mean train score by target ensemble size")
  #geom_jitter(aes(y = data__keyword, x = score, color = algo_hp), alpha=0.5, width=0, height=0.1)+

df %>% 
  pivot_longer(cols = c(test_scores, train_scores), values_to="score", names_to = "score_type") %>% 
  filter(data__cov_matrix=="identity",
         data__num_features==30,
         data__num_samples==5000,
         target__n_interactions==10,
         target__variant=="transform_sum",
         target__ensemble_size < 1.5) %>% 
  mutate(module__dropout_prob = if_else(is.na(module__dropout_prob), 0, module__dropout_prob)) %>% 
  mutate(module__dropout_prob = as_factor(module__dropout_prob)) %>% 
  group_by(method_name, target__ensemble_size, score_type, module__dropout_prob) %>% 
  mutate(mean_score = mean(score)) %>% 
  filter(score_type == "test_scores") %>% 
  ggplot() +
  geom_jitter(aes(x = target__ensemble_size, y = score, color = method_name), alpha=0.5, size=1)+
  geom_line(aes(x = target__ensemble_size, y = mean_score, color = method_name, linetype=module__dropout_prob))
#geom_jitter(aes(y = data__keyword, x = score, color = algo_hp), alpha=0.5, width=0, height=0.1)+

#RF MODIFIED

df <- #read_csv("results/new/sparse_in_interaction_gpu_06_01_128.csv") %>% 
  read_csv("results/new/sparse_interaction_cpu_07_01.csv") %>% 
              mutate(n_features_per_tree=1.) %>% 
  bind_rows(read_csv("results/new/sparse_interaction_cpu_07_01_rf_modified.csv"))

df %>% 
  pivot_longer(cols = c(test_scores, train_scores), values_to="score", names_to = "score_type") %>% 
  filter(data__cov_matrix=="identity",
         data__num_features==30,
         data__num_samples==5000,
         target__n_interactions==10,
         target__variant=="transform_sum",
         target__ensemble_size < 1.5) %>% 
  mutate(n_features_per_tree = as_factor(n_features_per_tree)) %>% 
  group_by(method_name, target__ensemble_size, score_type, n_features_per_tree) %>% 
  mutate(mean_score = mean(score)) %>% 
  filter(score_type == "test_scores") %>% 
  ggplot() +
  geom_jitter(aes(x = target__ensemble_size, y = score, color = method_name), alpha=0.2, size=1)+
  geom_point(aes(x = target__ensemble_size, y = mean_score, color = method_name, shape=n_features_per_tree), size=2)
#geom_jitter(aes(y = data__keyword, x = score, color = algo_hp), alpha=0.5, width=0, height=0.1)+

df %>% 
  pivot_longer(cols = c(test_scores, train_scores), values_to="score", names_to = "score_type") %>% 
  filter(data__cov_matrix=="identity",
         data__num_features==30,
         data__num_samples==5000,
         target__n_interactions==10,
         target__variant=="transform_sum",
         target__ensemble_size < 1.5) %>% 
  mutate(module__dropout_prob = if_else(is.na(module__dropout_prob), 0, module__dropout_prob)) %>% 
  mutate(module__dropout_prob = as_factor(module__dropout_prob)) %>% 
  group_by(method_name, target__ensemble_size, score_type, module__dropout_prob) %>% 
  mutate(mean_score = mean(score)) %>% 
  filter(score_type == "test_scores") %>% 
  ggplot() +
  geom_jitter(aes(x = target__ensemble_size, y = score, color = method_name), alpha=0.5, size=1)+
  geom_line(aes(x = target__ensemble_size, y = mean_score, color = method_name, linetype=module__dropout_prob))
#geom_jitter(aes(y = data__keyword, x = score, color = algo_hp), alpha=0.5, width=0, height=0.1)+


