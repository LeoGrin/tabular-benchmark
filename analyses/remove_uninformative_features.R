library(tidyverse)


df <- read_csv("results/sweeps/sweeps_classif/remove_useless/resnet_useless.csv") %>% 
  bind_rows(read_csv("results/sweeps/sweeps_classif/remove_useless/rf_useless.csv")) %>% 
  bind_rows(read_csv("results/sweeps/sweeps_classif/remove_useless/gpt_useless.csv")) %>% 
  bind_rows(read_csv("results/sweeps/sweeps_classif/remove_useless/ft_transformer_useless_features.csv")) %>% 
  bind_rows(read_csv("results/sweeps/sweeps_classif/remove_useless/useless_features_bonus.csv")) %>% 
  mutate(hp = "random") %>% 
  bind_rows(read_csv("results/sweeps/sweeps_classif/remove_useless/useless_features_default.csv") %>% 
              mutate(hp="default")) %>% 
  filter(transform__0__model_to_use == "rf_c") %>% 
  filter(!is.na(mean_val_score), !is.na(mean_test_score), !is.na(mean_time))



#################

res_datasets <- 
  df %>% 
  filter(!is.na(mean_time)) %>% 
  rename() %>% 
  normalize(variable=transform__0__num_features_to_remove, normalization_type = "quantile", quantile=0.1) %>% 
  random_search(variable =transform__0__num_features_to_remove, n_shuffles=30, default_first = T)


library(directlabels)
library(ggrepel)
library(shadowtext)

res_datasets_ <- res_datasets %>% 
  filter(random_rank == 30) %>%
  filter(data__keyword != "poker", data__keyword != "jannis") %>% 
  group_by(model_name, transform__0__num_features_to_remove, n_dataset) %>% 
  # mean on shuffles
  # mean will be NA if at least one score for one dataset is missing for a given time bin
  summarise(mean_test_score = mean(mean_test_score, na.rm=T)) %>% 
  summarise(test_score_high = max(mean_test_score), 
            test_score_low = min(mean_test_score),
            mean_test_score = mean(mean_test_score)) %>% 
  ungroup()
  #mutate(transform__0__num_features_to_remove = as_factor(100 * transform__0__num_features_to_remove)) %>% 
  #mutate(transform__0__num_features_to_remove = paste0(transform__0__num_features_to_remove, "%")) %>% 
res_datasets_ %>% 
  ggplot() +
  geom_ribbon(aes(x = transform__0__num_features_to_remove, ymin=test_score_low, ymax=test_score_high, fill=model_name), alpha=0.4) +
  geom_line(aes(x = transform__0__num_features_to_remove, y=mean_test_score, color=model_name, group=model_name), size=3) +
  geom_text_repel(aes(label=model_name,
                      color = model_name,
                      x = transform__0__num_features_to_remove,
                      y =  mean_test_score),
                  data= (res_datasets_ %>%
                           filter(transform__0__num_features_to_remove == 0.2)),
                  bg.color='white', size = 7,bg.r=0.15,
                  nudge_y = 0.03, nudge_x = 0.1
                  , min.segment.length=100)+
  #geom_dl(aes(label = model_name, x=transform__0__num_features_to_remove, y=mean_test_score, color=model_name), method = list(dl.combine("smart.grid"), cex=1.8))  +
  scale_x_continuous(labels = scales::percent) +
  xlab("Percentage of features removed \n (in decreasing order of RF importance)") +
  ylab("Normalized GBT test score of \n best model (on valid set) after \n 30 random search iterations") +
  theme_minimal(base_size=22) +
  #theme(legend.position="bottom", legend.title=element_blank(), legend.text = element_text(size=22)) +
  theme(legend.position="none") +
  colScale

ggsave("analyses/plots/useless_features.jpg", width=7, height=6)
#ggsave("analyses/plots/random_search_classif_numerical.jpg", width=7, height=6)

#######################
# Dataset by dataset



# 
# res_datasets %>% 
#   filter(random_rank == 10) %>%
#   filter(data__keyword != "poker", data__keyword != "jannis") %>% 
#   group_by(model_name, transform__0__num_features_to_remove, random_rank, data__keyword) %>% 
#   # mean on shuffles
#   # mean will be NA if at least one score for one dataset is missing for a given time bin
#   summarise(mean_test_score = mean(mean_test_score, na.rm=T)) %>% 
#   mutate(transform__0__num_features_to_remove = as_factor(100 * transform__0__num_features_to_remove)) %>% 
#   mutate(transform__0__num_features_to_remove = paste0(transform__0__num_features_to_remove, "%")) %>% 
#   ggplot() +
#   geom_boxplot(aes(x = transform__0__num_features_to_remove, y=mean_test_score, fill=model_name))+
#   xlab("Percentage of features removed \n (in decreasing order of RF importance)") +
#   ylab("Normalized GBT test score of best model \n (on valid set) after 20 random search iterations") +
#   theme_minimal(base_size=22) +
#   theme(legend.position="bottom", legend.title=element_blank(), legend.text = element_text(size=22))

res_datasets <- 
  df %>% 
  filter(!is.na(mean_time)) %>% 
  rename() %>% 
  #normalize(variable=transform__0__num_features_to_remove, normalization_type = "quantile", quantile=0.1) %>% 
  random_search(variable =transform__0__num_features_to_remove, n_shuffles=30, default_first = T)



res_datasets %>% 
  filter(random_rank == 30) %>%
  filter(data__keyword != "poker", data__keyword != "jannis") %>% 
  group_by(model_name, transform__0__num_features_to_remove, data__keyword) %>% 
  # mean on shuffles
  # mean will be NA if at least one score for one dataset is missing for a given time bin
  summarise(max_test_score=max(mean_test_score, na.rm=T),
            min_test_score=min(mean_test_score, na.rm=T),
            mean_test_score = mean(mean_test_score, na.rm=T)) %>% 
  #summarise(test_score_high = quantile(mean_test_score, 0.75), 
  #          test_score_low = quantile(mean_test_score, 0.25),
  #          mean_test_score = mean(mean_test_score)) %>% 
  ungroup() %>% 
  #mutate(transform__0__num_features_to_remove = as_factor(100 * transform__0__num_features_to_remove)) %>% 
  #mutate(transform__0__num_features_to_remove = paste0(transform__0__num_features_to_remove, "%")) %>% 
  ggplot() +
  #geom_ribbon(aes(x = transform__0__num_features_to_remove, ymin=test_score_low, ymax=test_score_high, fill=model_name), alpha=0.4) +
  geom_line(aes(x = transform__0__num_features_to_remove, y=mean_test_score, color=model_name), size=2) +
  geom_ribbon(aes(x = transform__0__num_features_to_remove, ymax=max_test_score, ymin=min_test_score, fill=model_name), alpha=0.3) +
  #geom_dl(aes(label = model_name, x=transform__0__num_features_to_remove, y=mean_test_score, color=model_name), method = list(dl.combine("smart.grid"), cex=1.8))  +
  scale_x_continuous(labels = scales::percent) +
  xlab("Percentage of features removed \n (in decreasing order of RF importance)") +
  ylab("Test accuracy of best \n model (on valid set) after \n 20 random search iterations") +
  theme_minimal(base_size=22) +
  facet_wrap(~data__keyword, scales = "free")+
  theme(legend.position="bottom", legend.title=element_blank(), legend.text = element_text(size=22)) +
  #theme(legend.position="none")# +
  colScale





ggsave("analyses/plots/useless_features_datasets.jpg", width=15, height=9)
























































# 
# 
# res_datasets %>% 
#   filter(random_rank == 20) %>%
#   #filter(model_name == "GradientBoostingTree") %>%
#   filter(data__keyword != "poker") %>% 
#   group_by(model_name, data__keyword, transform__0__num_features_to_remove) %>% 
#   summarise(mean_test_score = mean(mean_test_score)) %>% 
#   mutate(transform__0__num_features_to_remove = as_factor(100 * transform__0__num_features_to_remove)) %>% 
#   mutate(transform__0__num_features_to_remove = paste0(transform__0__num_features_to_remove, "%")) %>% 
#   group_by(transform__0__num_features_to_remove, model_name) %>% 
#   summarise(mean_test_score_median = mean(mean_test_score, na.rm=T), mean_test_score_high = quantile(mean_test_score, 0.90, na.rm=T), mean_test_score_low = quantile(mean_test_score, 0.10, na.rm=T)) %>% 
#   #mutate(transform__0__num_features_to_remove = as.character(transform__0__num_features_to_remove)) %>% 
#   #mutate(model_name = fct_relevel(model_name, c("RandomForest", "GradientBoostingTree", "FT_Transformer"))) %>%
#   ggplot() +
#   #geom_boxplot(aes(x = transform__0__num_features_to_remove, y=mean_test_score, fill=model_name))+
#   geom_ribbon(aes(x = transform__0__num_features_to_remove, ymin = mean_test_score_low, ymax=mean_test_score_high, fill=model_name, group=model_name), alpha=0.3) +
#   geom_line(aes(x = transform__0__num_features_to_remove, y=mean_test_score_median, color=model_name, group=model_name), size=3) 
#   #geom_dl(aes(label = model_name, x=transform__0__num_features_to_remove, y=mean_test_score, color=model_name), method = list(dl.combine("smart.grid"), cex=1.3))  +
#   xlab("Percentage of features removed \n (in decreasing order of RF importance)") +
#   ylab("Normalized GBT test score of best model \n (on valid set) after 20 random search iterations") +
#   theme_minimal(base_size=22) +
#   theme(legend.position="bottom", legend.title=element_blank(), legend.text = element_text(size=22))
# 
# res_datasets %>% 
#   filter(random_rank == 20) %>%
#   #filter(model_name == "GradientBoostingTree") %>%
#   filter(data__keyword != "poker") %>% 
#   group_by(model_name, data__keyword, transform__0__num_features_to_remove) %>% 
#   summarise(mean_test_score = mean(mean_test_score)) %>% 
#   mutate(transform__0__num_features_to_remove = as_factor(100 * transform__0__num_features_to_remove)) %>% 
#   mutate(transform__0__num_features_to_remove = paste0(transform__0__num_features_to_remove, "%")) %>% 
#   #mutate(transform__0__num_features_to_remove = as.character(transform__0__num_features_to_remove)) %>% 
#   #mutate(model_name = fct_relevel(model_name, c("RandomForest", "GradientBoostingTree", "FT_Transformer"))) %>%
#   ggplot() +
#   geom_point(aes(x = transform__0__num_features_to_remove, y =mean_test_score, color = data__keyword)) +
#   #geom_boxplot(aes(x = transform__0__num_features_to_remove, y=mean_test_score, fill=model_name))+
#   #geom_dl(aes(label = method, x=transform__0__num_features_to_remove, y=mean_test_score, color=method), method = list(dl.combine("smart.grid"), cex=1.3))  +
#   facet_wrap(~model_name, ncol=1) +
#   xlab("Percentage of features removed \n (in decreasing order of RF importance)") +
#   ylab("Normalized GBT test score of best model \n (on valid set) after 20 random search iterations") +
#   theme_minimal(base_size=22) +
#   
#   #theme(legend.position="bottom", legend.title=element_blank(), legend.text = element_text(size=22))
# 
# 
# ggsave("analyses/plots/removed_features_2.jpg", width=16, height=9)
# 
# 
# 
# ggplot() +
#   #geom_hline(aes(yintercept=mean_test_score, color=model_name), data=res_ %>% filter(random_rank == 0), linetype="dotted", size=1.5) +
#   geom_line(aes(x = random_rank, y = mean_test_score, color = transform__0__num_features_to_remove, group = transform__0__num_features_to_remove), data=res_, size=2) +
#   geom_dl(aes(label = transform__0__num_features_to_remove, x=random_rank, y=mean_test_score, color=transform__0__num_features_to_remove), method = list(dl.combine("smart.grid"), cex=1.3), data=res_)  +
#   #geom_ribbon(aes(x=random_rank, ymin = min_test_score, ymax=max_test_score, fill=transform__0__num_features_to_remove), data=res_, alpha=0.3) +
#   facet_wrap(~model_name)+
#   #geom_text(aes(x=150, y=mean_test_score + 0.0007, color=model_name, label=str_c(model_name, " (Default)")), data=res_ %>% filter(random_rank == 0), size=7) +
#   scale_x_log10() +
#   xlab("Number of random search iterations") +
#   ylab("Normalized test score of best model \n (on valid set) up to this iteration") +
#   theme_minimal(base_size=22)+
#   theme(legend.position="none")
# 
# ggsave("analyses/plots/removed_features.jpg", width=16, height=9)
# 
# # plot the difference between score vs difference for gbt
# 
# t <- res_ %>%
#   select(-max_test_score, -min_test_score) %>% 
#   slice_tail(n=1) %>% 
#   group_by(model_name) %>% 
#   pivot_wider(names_from=transform__0__num_features_to_remove, values_from=mean_test_score) %>% 
#   pivot_longer(cols = c("0.01", "0.02", "0.05"), names_to = "threshold", values_to="mean_test_score")
# 
# 
# new_tibble <- tibble(gap=(t$"0" - t$mean_test_score) / t$"0", max_rel_decrease=t$threshold, model=t$model_name)
# 
# 
# new_tibble %>% 
#   mutate(max_rel_decrease = as.numeric(max_rel_decrease)) %>% 
#   ggplot() +
#   geom_line(aes(x = max_rel_decrease, y = gap, color=model), size=2) +
#   geom_line(aes(x=max_rel_decrease, y=max_rel_decrease, size=2)) +
#   geom_dl(aes(label = model, x=max_rel_decrease, y=gap, color=model), method = list(dl.combine("smart.grid"), cex=1.3))  +
#   xlab("Maximum relative decrease (GBT)") +
#   ylab("Actual decrease") +
#   theme_minimal(base_size=22)+
#   theme(legend.position="none")
# 
# ggplot() +
#   geom_point(aes(x = transform__0__num_features_to_remove, y = ))
# 
# 
# res_ %>%
#   select(-max_test_score, -min_test_score) %>% 
#   slice_tail(n=1)
# 
# # ggplot() +
# #   geom_line(aes(x = random_rank, y = mean_test_score, color=model_name, group=model_name), data=res_, size=2) +
# #   geom_dl(aes(label = model_name, x=random_rank, y=mean_test_score, color=model_name), method = list(dl.combine("smart.grid"), cex=1.3), data=res_)  +
# #   #geom_hline(aes(yintercept=mean_test_score, color=model_name), data=res_ %>% filter(max_rank == 0), linetype="dotted", size=1.5) +
# #   #geom_line(aes(x = log_cum_time_factor, y = max_test_score, color=model_name, group=model_name)) +
# #   #geom_line(aes(x = log_cum_time_factor, y = min_test_score, color=model_name, group=model_name)) +
# #   geom_ribbon(aes(x = random_rank, ymin=min_test_score, ymax=max_test_score, fill=model_name), data=res_, alpha=0.2) +
# #   #geom_point(aes(x = cum_time, y = mean_test_score, color=model_name, group=model_name)) +
# #   #scale_x_log10(limits= c(10, 1E5)) +
# #   scale_x_log10() +
# #   #ylim(0.977, 0.993)+
# #   xlab("Random search time (seconds)") +
# #   ylab("Normalized test score of best model \n (on valid set) up to this time") +
# #   theme_minimal(base_size=22) +
# #   theme(legend.position="none")
# 
# 
# res_time_ <- res_time %>% #select(model_name, data__keyword, log_cum_time_factor,log_cum_time, mean_val_score,best_val_score_so_far, mean_test_score) %>% 
#   group_by(model_name, cum_time_factor) %>% 
#   summarise(count=sum(!is.na(mean_test_score)),
#             max_test_score = mean(mean_test_score, na.rm=T) + 2 * sd(mean_test_score, na.rm=T) / sqrt(count),
#             min_test_score = mean(mean_test_score, na.rm=T) - 2 * sd(mean_test_score, na.rm=T) / sqrt(count),
#             mean_test_score = mean(mean_test_score, na.rm=T),
#             cum_time = mean(cum_time))
# 
# 
# library(directlabels)
# 
# ggplot() +
#   geom_line(aes(x = cum_time, y = mean_test_score, color=model_name, group=model_name), data=res_time_, size=2) +
#   geom_dl(aes(label = model_name, x=cum_time, y=mean_test_score, color=model_name), method = list(dl.combine("smart.grid"), cex=1.3), data=res_time_)  +
#   #geom_hline(aes(yintercept=mean_test_score, color=model_name), data=res_ %>% filter(max_rank == 0), linetype="dotted", size=1.5) +
#   #geom_line(aes(x = log_cum_time_factor, y = max_test_score, color=model_name, group=model_name)) +
#   #geom_line(aes(x = log_cum_time_factor, y = min_test_score, color=model_name, group=model_name)) +
#   geom_ribbon(aes(x = cum_time, ymin=min_test_score, ymax=max_test_score, fill=model_name), data=res_time_, alpha=0.2) +
#   #geom_point(aes(x = cum_time, y = mean_test_score, color=model_name, group=model_name)) +
#   scale_x_log10(limits= c(10, 1E5)) +
#   #ylim(0.977, 0.993)+
#   xlab("Random search time (seconds)") +
#   ylab("Normalized test score of best model \n (on valid set) up to this time") +
#   theme_minimal(base_size=22) +
#   theme(legend.position="none")# +
# #scale_x_discrete(breaks=c(10, 60, 3600, 3600*24), labels=c("10 seconds", "1 minute", "1 hour", "1 day"))
# 
# ggsave("analyses/plots/random_search_time.jpg", width=16, height=9)
# 
# 
# # Iters
# 
# #TODO: think per dataset?
# res_ <- res %>% 
#   group_by(model_name, max_rank) %>% 
#   summarise(max_test_score = mean(mean_test_score) + 2 * sd(mean_test_score, na.rm=T) / sqrt(N_SHUFFLE),
#             min_test_score = mean(mean_test_score) - 2 * sd(mean_test_score, na.rm=T) / sqrt(N_SHUFFLE),
#             mean_test_score = mean(mean_test_score)) 
# ggplot() +
#   geom_hline(aes(yintercept=mean_test_score, color=model_name), data=res_ %>% filter(max_rank == 0), linetype="dotted", size=1.5) +
#   geom_line(aes(x = max_rank, y = mean_test_score, color = model_name), data=res_, size=2) +
#   geom_ribbon(aes(x=max_rank, ymin = min_test_score, ymax=max_test_score, fill=model_name), data=res_, alpha=0.3) +
#   geom_text(aes(x=150, y=mean_test_score + 0.0004, color=model_name, label=str_c(model_name, " (Default)")), data=res_ %>% filter(max_rank == 0), size=7) +
#   scale_x_log10() +
#   xlab("Number of random search iterations") +
#   ylab("Normalized test score of best model \n (on valid set) up to this iteration") +
#   theme_minimal(base_size=22) +
#   theme(legend.position="none")
# 
# 
# 
# 
# 
# 
# 
