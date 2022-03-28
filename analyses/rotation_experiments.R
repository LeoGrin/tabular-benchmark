df <- read_csv("results/rotation_experiment.csv")


df %>% 
  #filter(transform__0__importance_cutoff <= 1.) %>% 
  pivot_longer(cols = c(test_scores, train_scores), values_to="score", names_to = "score_type") %>% 
  #group_by(method_name, score_type, data_generation_str, model_params_str, 
  #         transform__0__method_name, transform__0__importance_cutoff, data__keyword) %>% 
  #summarise(mean_score_iter = mean(score, na.rm=T)) %>% 
  #filter(data__num_samples %in% c(500, 1000, 5000)) %>% 
  #mutate(data__num_samples = as_factor(data__num_samples)) %>% 
  ggplot() +
  geom_jitter(aes(x = score, y = data__keyword, color=transform__0__method_name, group=method_name), alpha=0.5, width=0, height=0.1) +
  facet_wrap(method_name~score_type)


df <- read_csv("results/rotation_experiment2.csv")


df %>% 
  #filter(transform__0__importance_cutoff <= 1.) %>% 
  pivot_longer(cols = c(test_scores, train_scores), values_to="score", names_to = "score_type") %>% 
  #group_by(method_name, score_type, data_generation_str, model_params_str, 
  #         transform__0__method_name, transform__0__importance_cutoff, data__keyword) %>% 
  #summarise(mean_score_iter = mean(score, na.rm=T)) %>% 
  #filter(data__num_samples %in% c(500, 1000, 5000)) %>% 
  #mutate(data__num_samples = as_factor(data__num_samples)) %>% 
  ggplot() +
  geom_jitter(aes(x = score, y = data__keyword, color=transform__0__method_name, group=method_name), alpha=0.5, width=0, height=0.1) +
  facet_wrap(method_name~score_type)

df %>% 
  #filter(transform__0__importance_cutoff <= 1.) %>% 
  pivot_longer(cols = c(test_scores, train_scores), values_to="score", names_to = "score_type") %>% 
  filter(score_type == "test_scores") %>% 
  group_by(method_name, score_type, data_generation_str, model_params_str, data__keyword, transform__0__method_name) %>% 
  summarise(mean_score_iter = mean(score, na.rm=T)) %>% 
  #filter(data__num_samples %in% c(500, 1000, 5000)) %>% 
  #mutate(data__num_samples = as_factor(data__num_samples)) %>% 
  ungroup() %>% 
  ggplot() +
  geom_point(aes(x = mean_score_iter, y = data__keyword, shape=transform__0__method_name, color=method_name=="mlp", group = transform__0__method_name), position = position_dodge2(width=0.5))+
  xlab("Test score (25 runs)") +
  ylab("Dataset")
  #facet_wrap(~score_type)

df %>% 
  #filter(transform__0__importance_cutoff <= 1.) %>% 
  filter(data__keyword != "mnist_1_7") %>% 
  pivot_longer(cols = c(test_scores, train_scores), values_to="score", names_to = "score_type") %>% 
  filter(score_type == "test_scores") %>% 
  group_by(method_name, score_type, model_params_str, transform__0__method_name) %>% 
  summarise(mean_score = mean(score, na.rm=T), sd_score=sd(score, na.rm=T) / sqrt(n())) %>% 
  #filter(data__num_samples %in% c(500, 1000, 5000)) %>% 
  #mutate(data__num_samples = as_factor(data__num_samples)) %>% 
  ungroup() %>% 
  ggplot(aes(x=transform__0__method_name)) +
  geom_bar(aes(y=mean_score, fill=method_name), stat = "identity", position=position_dodge(), alpha=0.8)+
  geom_errorbar(aes(ymin=mean_score-sd_score, ymax=mean_score+sd_score, color = method_name), width=.2,
                position=position_dodge(.9))  + 
  xlab("Test score (25 runs)") +
  ylab("Dataset")
#facet_wrap(~score_type)

df %>% 
  #filter(transform__0__importance_cutoff <= 1.) %>% 
  pivot_longer(cols = c(test_scores, train_scores), values_to="score", names_to = "score_type") %>% 
  filter(score_type == "test_scores") %>% 
  group_by(method_name, score_type, data_generation_str, model_params_str, data__keyword, transform__0__method_name) %>% 
  summarise(mean_score_iter = mean(score, na.rm=T)) %>% 
  #group_by(method_name, score_type, model_params_str, transform__0__method_name) %>% 
  #summarise(mean_score = mean(score, na.rm=T), sd_score=sd(score, na.rm=T) / sqrt(n())) %>% 
  #filter(data__num_samples %in% c(500, 1000, 5000)) %>% 
  #mutate(data__num_samples = as_factor(data__num_samples)) %>% 
  ungroup() %>% 
  ggplot(aes(x=mean_score_iter, shape=method_name)) +
  geom_point(aes(color = data__keyword, y=method_name)) +
  facet_wrap(~transform__0__method_name)
  xlab("Test score (25 runs)") +
  ylab("Dataset") 


df %>% 
  #filter(transform__0__importance_cutoff <= 1.) %>% 
  pivot_longer(cols = c(test_scores, train_scores), values_to="score", names_to = "score_type") %>% 
  group_by(method_name, score_type, data_generation_str, model_params_str, data__keyword, transform__0__method_name) %>% 
  summarise(mean_score_iter = mean(score, na.rm=T)) %>% 
  #filter(data__num_samples %in% c(500, 1000, 5000)) %>% 
  #mutate(data__num_samples = as_factor(data__num_samples)) %>% 
  pivot_wider(names_from=transform__0__method_name, values_from=mean_score_iter) %>% 
  group_by(data__keyword) %>%
  fill(everything(), .direction = "down") %>%
  fill(everything(), .direction = "up") %>%
  distinct() %>% 
  ggplot() +
  geom_segment(aes(x = no_transform, xend=random_rotation, y = data__keyword, yend=data__keyword, color=method_name, group=method_name), position = position_dodge2(width=0.2)) +
  facet_wrap(~score_type)


# removing features

df <- read_csv("results/rotation_experiment_features_removed.csv")


df %>% 
  filter(method_name == "hgbt") %>% 
  #filter(transform__0__importance_cutoff <= 1.) %>% 
  pivot_longer(cols = c(test_scores, train_scores), values_to="score", names_to = "score_type") %>% 
  filter(score_type == "test_scores") %>% 
  #group_by(method_name, score_type, data_generation_str, model_params_str, 
  #         transform__0__method_name, transform__0__importance_cutoff, data__keyword) %>% 
  #summarise(mean_score_iter = mean(score, na.rm=T)) %>% 
  #filter(data__num_samples %in% c(500, 1000, 5000)) %>% 
  #mutate(data__num_samples = as_factor(data__num_samples)) %>% 
  ggplot() +
  geom_jitter(aes(x = score, y = data__keyword, color=transform__1__method_name, group=method_name), alpha=0.5, width=0, height=0.1) +
  facet_wrap(~transform__0__num_features_to_remove)

df %>% 
  #filter(transform__0__importance_cutoff <= 1.) %>% 
  pivot_longer(cols = c(test_scores, train_scores), values_to="score", names_to = "score_type") %>% 
  filter(score_type == "test_scores") %>% 
  group_by(method_name, score_type, data_generation_str, model_params_str, data__keyword, transform__1__method_name, transform__0__num_features_to_remove) %>% 
  summarise(mean_score_iter = mean(score, na.rm=T)) %>% 
  #filter(data__num_samples %in% c(500, 1000, 5000)) %>% 
  #mutate(data__num_samples = as_factor(data__num_samples)) %>% 
  ungroup() %>% 
  ggplot() +
  geom_point(aes(x = mean_score_iter, y = data__keyword, shape=transform__1__method_name, color=method_name=="mlp", group = transform__1__method_name), position = position_dodge2(width=0.5))+
  facet_wrap(~transform__0__num_features_to_remove)+
  xlab("Test score (25 runs)") +
  ylab("Dataset")

# does MLP works better without useless features

df %>% 
  #filter(transform__0__importance_cutoff <= 1.) %>% 
  filter(transform__1__method_name == "no_transform") %>% 
  pivot_longer(cols = c(test_scores, train_scores), values_to="score", names_to = "score_type") %>% 
  filter(score_type == "test_scores") %>% 
  group_by(method_name, score_type, data_generation_str, model_params_str, data__keyword, transform__1__method_name, transform__0__num_features_to_remove) %>% 
  summarise(mean_score_iter = mean(score, na.rm=T)) %>% 
  #mutate(transform__0__num_features_to_remove = as.character(transform__0__num_features_to_remove)) %>% 
  #filter(data__num_samples %in% c(500, 1000, 5000)) %>% 
  #mutate(data__num_samples = as_factor(data__num_samples)) %>% 
  ungroup() %>% 
  ggplot() +
  geom_point(aes(x = mean_score_iter, y = data__keyword, shape=method_name=="mlp", color=transform__0__num_features_to_remove, group = transform__0__num_features_to_remove), position = position_dodge2(width=0.5))+
  #facet_wrap(~transform__0__num_features_to_remove)+
  xlab("Test score (25 runs)") +
  ylab("Dataset")
#facet_wrap(~score_type)
