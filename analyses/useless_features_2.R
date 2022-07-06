library(tidyverse)


#df <- read_csv("results/sweeps/remove_features/resnet_remove_features.csv") %>% 
#  bind_rows(read_csv("results/sweeps/remove_features/ft_transformer_remove_features.csv")) %>% 
##  bind_rows(read_csv("results/sweeps/sweep_resnet.csv") %>% 
#             mutate(transform__0__num_features_to_remove = 0) %>% 
#             slice_head(n=1000)) %>% 
# bind_rows(read_csv("results/sweeps/sweep_ft_transformer.csv") %>% 
#             mutate(transform__0__num_features_to_remove = 0) %>% 
#             slice_head(n=1000)) %>% 
# mutate(transform__0__num_features_to_remove = as_factor(transform__0__num_features_to_remove)) %>% 
# filter(data__keyword %in% c("heloc", "electricity", "california", "covtype", "churn", "cpu", "wine"))


df <- read_csv("results/sweeps/sweeps_classif/remove_useless/rf_useless.csv") %>% 
  bind_rows(read_csv("results/sweeps/sweeps_classif/remove_useless/gpt_useless.csv")) %>% 
  filter(transform__0__model_to_use == "rf_c") %>% 
  filter(!is.na(mean_val_score), !is.na(mean_test_score), !is.na(mean_time))
  
df <- read_csv("results/sweeps/sweeps_classif/remove_useless/gpt_useless_keep.csv") %>%
  mutate(method = "keep") %>% 
  bind_rows(read_csv("results/sweeps/sweeps_classif/remove_useless/gpt_useless.csv") %>% 
              mutate(method = "remove")) %>% 
  filter(transform__0__model_to_use == "rf_c") %>% 
  filter(!is.na(mean_val_score), !is.na(mean_test_score), !is.na(mean_time))



#################

df_normalized <- df %>% 
  mutate(model_name = case_when(
    model_name == "rf_c" ~ "RandomForest",
    model_name == "xgb_c" ~ "XGBoost",
    model_name == "gbt_c" ~ "GradientBoostingTree",
    model_name == "ft_transformer" ~ "FT_Transformer",
    model_name == "rtdl_resnet" ~ "Resnet")) %>% 
  filter(!is.na(mean_time)) %>% 
  group_by(data__keyword) %>% #relative to best score on dataset (with all features)
  #mutate(mean_val_score = mean_val_score / max(mean_val_score, na.rm=T), mean_test_score = mean_test_score / max(mean_test_score, na.rm=T)) %>% 
  #mutate(mean_val_score = (mean_val_score - quantile(mean_val_score, 0.05)) / (max(mean_val_score, na.rm=T) - quantile(mean_val_score, 0.05)), mean_test_score = (mean_test_score - quantile(mean_test_score, 0.05)) / (max(mean_test_score, na.rm=T) - quantile(mean_test_score, 0.05))) %>% 
  mutate(mean_val_score = (mean_val_score - 0.5) / (max(mean_val_score, na.rm=T) - 0.5), mean_test_score = (mean_test_score - 0.5) / (max(mean_test_score, na.rm=T) - 0.5)) %>% 
  select(model_name, data__keyword, mean_val_score, mean_test_score, mean_time, transform__0__num_features_to_remove, method)


# ITERS

library(zoo)

N_SHUFFLE <- 20

res <- tibble()

for (i in 1:N_SHUFFLE) {
  new_df <- df_normalized %>% 
    add_column(random_number = runif(nrow(.))) %>% 
    group_by(model_name, data__keyword, transform__0__num_features_to_remove) %>% 
    mutate(random_rank = rank(random_number))%>% #default hp as first iter 
    select(-random_number) %>% 
    arrange(random_rank)
  
  res <- res %>%  bind_rows(new_df %>% 
                              #filter(hp == "random") %>% 
                              mutate(best_val_score_so_far = cummax(mean_val_score)) %>% 
                              mutate(mean_test_score = if_else(mean_val_score == best_val_score_so_far, mean_test_score, NA_real_)) %>% 
                              #filter(mean_val_score == best_val_score_so_far) %>% 
                              # Only keep the first time we beat valid score
                              group_by(model_name, data__keyword, transform__0__num_features_to_remove, mean_val_score) %>% 
                              #filter(random_rank == min(random_rank)) %>%
                              mutate(mean_test_score = if_else(random_rank == min(random_rank), mean_test_score, NA_real_)) %>% 
                              ungroup(mean_val_score) %>% 
                              #for every dataset, prolong the curve by the last best value when there is no value left (for long time)
                              ungroup() %>% 
                              complete(model_name, data__keyword, transform__0__num_features_to_remove, random_rank) %>%
                              #select(model_name, data__keyword, log_cum_time_factor,log_cum_time, mean_val_score,best_val_score_so_far, mean_test_score)) %>% 
                              group_by(model_name, transform__0__num_features_to_remove, data__keyword) %>% 
                              #mutate(mean_val_score = na.locf(mean_val_score, na.rm=F)) %>% 
                              mutate(mean_test_score = na.locf(mean_test_score, na.rm=F))%>% 
                              #filter(model_name == "Resnet") %>% 
                              #select(model_name, data__keyword, log_cum_time_factor,log_cum_time, mean_val_score,best_val_score_so_far, mean_test_score) %>% 
                              ungroup() %>% 
                              group_by(model_name, transform__0__num_features_to_remove, random_rank) %>% 
                              # mean on datasets
                              # mean will be NA if at least one score for one dataset is missing for a given time bin
                              summarise( mean_test_score = mean(mean_test_score)) %>% 
                              # Take log cum time as the mean of the bin on all models
                              mutate(n_dataset = i))
}

# ITERS on datasets

N_SHUFFLE <- 30

res_datasets <- tibble()

for (i in 1:N_SHUFFLE) {
  new_df <- df_normalized %>% 
    add_column(random_number = runif(nrow(.))) %>% 
    group_by(model_name, data__keyword, transform__0__num_features_to_remove, method) %>% 
    #mutate(random_rank = if_else(hp=="default", 0, rank(random_number)))%>% #default hp as first iter 
    mutate(random_rank = rank(random_number))%>%
    select(-random_number) %>% 
    arrange(random_rank)
  
  res_datasets <- res_datasets %>%  bind_rows(new_df %>% 
                                                #filter(hp == "random") %>% 
                                                mutate(best_val_score_so_far = cummax(mean_val_score)) %>% 
                                                mutate(mean_test_score = if_else(mean_val_score == best_val_score_so_far, mean_test_score, NA_real_)) %>% 
                                                #filter(mean_val_score == best_val_score_so_far) %>% 
                                                # Only keep the first time we beat valid score
                                                group_by(model_name, data__keyword, transform__0__num_features_to_remove, method, mean_val_score) %>% 
                                                #filter(random_rank == min(random_rank)) %>%
                                                mutate(mean_test_score = if_else(random_rank == min(random_rank), mean_test_score, NA_real_)) %>% 
                                                ungroup(mean_val_score) %>% 
                                                #for every dataset, prolong the curve by the last best value when there is no value left (for long time)
                                                ungroup() %>% 
                                                complete(model_name, data__keyword, transform__0__num_features_to_remove, method, random_rank) %>%
                                                #select(model_name, data__keyword, log_cum_time_factor,log_cum_time, mean_val_score,best_val_score_so_far, mean_test_score)) %>% 
                                                group_by(model_name, transform__0__num_features_to_remove, method, data__keyword) %>% 
                                                #mutate(mean_val_score = na.locf(mean_val_score, na.rm=F)) %>% 
                                                mutate(mean_test_score = na.locf(mean_test_score, na.rm=F))%>% 
                                                #filter(model_name == "Resnet") %>% 
                                                #select(model_name, data__keyword, log_cum_time_factor,log_cum_time, mean_val_score,best_val_score_so_far, mean_test_score) %>% 
                                                ungroup() %>% 
                                               # group_by(model_name, random_rank, transform__0__num_features_to_remove) %>% 
                                                # mean on datasets
                                                # mean will be NA if at least one score for one dataset is missing for a given time bin
                                                #summarise( mean_test_score = mean(mean_test_score)) %>% 
                                                # Take log cum time as the mean of the bin on all models
                                                mutate(n_dataset = i))
}



res_ <- res %>% #select(model_name, data__keyword, log_cum_time_factor,log_cum_time, mean_val_score,best_val_score_so_far, mean_test_score) %>% 
  group_by(model_name, transform__0__num_features_to_remove, random_rank) %>% 
  summarise(count=sum(!is.na(mean_test_score)),
            max_test_score = mean(mean_test_score, na.rm=T) + 2 * sd(mean_test_score, na.rm=T) / sqrt(count),
            min_test_score = mean(mean_test_score, na.rm=T) - 2 * sd(mean_test_score, na.rm=T) / sqrt(count),
            mean_test_score = mean(mean_test_score, na.rm=T))
library(directlabels)

res_datasets %>% 
  filter(random_rank == 20) %>%
  filter(model_name == "GradientBoostingTree") %>%
  filter(data__keyword != "poker") %>% 
  group_by(model_name, data__keyword, transform__0__num_features_to_remove, method) %>% 
  summarise(mean_test_score = mean(mean_test_score)) %>% 
  mutate(transform__0__num_features_to_remove = as_factor(100 * transform__0__num_features_to_remove)) %>% 
  mutate(transform__0__num_features_to_remove = paste0(transform__0__num_features_to_remove, "%")) %>% 
  #mutate(transform__0__num_features_to_remove = as.character(transform__0__num_features_to_remove)) %>% 
  #mutate(model_name = fct_relevel(model_name, c("RandomForest", "GradientBoostingTree", "FT_Transformer"))) %>%
  mutate(method = case_when(
    method == "keep" ~ "Fit on removed features only",
    method == "remove" ~ "Fit without removed features")) %>% 
  ggplot() +
  #geom_line(aes(x = transform__0__num_features_to_remove, y =mean_test_score, color = data__keyword)) +
  geom_boxplot(aes(x = transform__0__num_features_to_remove, y=mean_test_score, fill = method))+
  #geom_dl(aes(label = method, x=transform__0__num_features_to_remove, y=mean_test_score, color=method), method = list(dl.combine("smart.grid"), cex=1.3))  +
  #facet_wrap(~method, ncol=1) +
  xlab("Percentage of features removed \n (in decreasing order of RF importance)") +
  ylab("Normalized GBT test score of best model \n (on valid set) after 20 random search iterations") +
  theme_minimal(base_size=22) +
  theme(legend.position="bottom", legend.title=element_blank(), legend.text = element_text(size=22))

res_datasets %>% 
  filter(random_rank == 20) %>%
  filter(model_name == "GradientBoostingTree") %>%
  filter(data__keyword != "poker") %>% 
  #group_by(model_name, data__keyword, transform__0__num_features_to_remove, method) %>% 
  #summarise(mean_test_score = mean(mean_test_score)) %>% 
  #mutate(transform__0__num_features_to_remove = as_factor(100 * transform__0__num_features_to_remove)) %>% 
  mutate(transform__0__num_features_to_remove = as_factor(100 * transform__0__num_features_to_remove)) %>% 
  #mutate(transform__0__num_features_to_remove = paste0(transform__0__num_features_to_remove, "%")) %>% 
  group_by(model_name, data__keyword, method, transform__0__num_features_to_remove) %>% 
  summarise(max_test_score = max(mean_test_score), min_test_score = min(mean_test_score), mean_test_score = mean(mean_test_score)) %>% 
  ungroup() %>% 
  #mutate(transform__0__num_features_to_remove = as.character(transform__0__num_features_to_remove)) %>% 
  #mutate(model_name = fct_relevel(model_name, c("RandomForest", "GradientBoostingTree", "FT_Transformer"))) %>%
  mutate(method = case_when(
    method == "keep" ~ "Fit on removed features only",
    method == "remove" ~ "Fit without removed features")) %>% 
  ggplot() +
  #geom_line(aes(x = transform__0__num_features_to_remove, y =mean_test_score, color = data__keyword)) +
  geom_ribbon(aes(x = transform__0__num_features_to_remove, ymax=max_test_score, ymin=min_test_score, fill = method, group=method), alpha=0.4) +
  geom_line(aes(x = transform__0__num_features_to_remove, y=mean_test_score, color=method, group=method), size=1.5) +
  #geom_boxplot(aes(x = transform__0__num_features_to_remove, y=mean_test_score, fill = method))+
  #geom_dl(aes(label = method, x=transform__0__num_features_to_remove, y=mean_test_score, color=method), method = list(dl.combine("smart.grid"), cex=1.3))  +
  #facet_wrap(~method, ncol=1) +
  facet_wrap(~data__keyword, scales="free") +
  scale_x_discrete(guide = guide_axis(check.overlap = TRUE))+
  #scale_x_discrete(guide = guide_axis(n.dodge=3))+
  xlab("Percentage of features removed \n (in decreasing order of RF importance)") +
  ylab("Normalized GBT test score of best model \n (on valid set) after 20 random search iterations") +
  theme_minimal(base_size=20) +
  theme(legend.position="bottom", legend.title=element_blank(), legend.text = element_text(size=22))

ggsave("analyses/plots/remove_features_datasets.jpg", width=15, height=9)



#TODO:remove dataset from bin where one value is missing

res_datasets_ <- res_datasets %>% 
  filter(random_rank == 20) %>%
  filter(model_name == "GradientBoostingTree") %>%
  filter(data__keyword != "poker") %>% 
  group_by(model_name, data__keyword, transform__0__num_features_to_remove, method) %>% 
  summarise(mean_test_score = mean(mean_test_score)) %>% 
  ungroup() %>% 
  group_by(model_name, transform__0__num_features_to_remove, method) %>% 
  mutate(test_score_high=quantile(mean_test_score, 0.9, na.rm=T), test_score_low=quantile(mean_test_score, 0.1, na.rm=T),
         mean_test_score=mean(mean_test_score, na.rm=T)) %>% 
  mutate(transform__0__num_features_to_remove = as_factor(100 * transform__0__num_features_to_remove)) %>% 
  mutate(transform__0__num_features_to_remove = paste0(transform__0__num_features_to_remove, "%")) %>% 
  #mutate(transform__0__num_features_to_remove = as.character(transform__0__num_features_to_remove)) %>% 
  #mutate(model_name = fct_relevel(model_name, c("RandomForest", "GradientBoostingTree", "FT_Transformer"))) %>%
  mutate(method = case_when(
    method == "keep" ~ "Fit on removed features only",
    method == "remove" ~ "Fit without removed features"))
res_datasets_ %>% 
  ggplot() +
  #geom_line(aes(x = transform__0__num_features_to_remove, y =mean_test_score, color = data__keyword)) +
  geom_ribbon(aes(x = transform__0__num_features_to_remove, ymax=test_score_high, ymin=test_score_low, fill = method, group=method), alpha=0.4) +
  geom_line(aes(x = transform__0__num_features_to_remove, y=mean_test_score, color = method, group=method), size=2)+
  geom_text_repel(aes(label=method, 
                      color = method,
                      x = transform__0__num_features_to_remove,
                      y =  mean_test_score),
                  data= (res_datasets_ %>% 
                           filter(data__keyword=="phoneme",
                                  transform__0__num_features_to_remove == "10%")),
                  bg.color='white', size = 6.5, bg.r=0.15,
                  nudge_y = -0.05, nudge_x = 0.1, min.segment.length=10,
                  max.overlaps=100)+
  #geom_dl(aes(label = method, x=transform__0__num_features_to_remove, y=mean_test_score, color=method), method = list(dl.combine("smart.grid"), cex=1.3))  +
  #facet_wrap(~method, ncol=1) +
  xlab("Percentage of features removed \n (in decreasing order of RF importance)") +
  ylab("Normalized GBT test score of \n best model (on valid set) after \n 20 random search iterations") +
  theme_minimal(base_size=22) +
  theme(legend.position="none")
  #theme(legend.position="bottom", legend.title=element_blank(), legend.text = element_text(size=22))
  

ggsave("analyses/plots/removed_features_2.jpg", width=7, height=6)

  

ggplot() +
  #geom_hline(aes(yintercept=mean_test_score, color=model_name), data=res_ %>% filter(random_rank == 0), linetype="dotted", size=1.5) +
  geom_line(aes(x = random_rank, y = mean_test_score, color = transform__0__num_features_to_remove, group = transform__0__num_features_to_remove), data=res_, size=2) +
  geom_dl(aes(label = transform__0__num_features_to_remove, x=random_rank, y=mean_test_score, color=transform__0__num_features_to_remove), method = list(dl.combine("smart.grid"), cex=1.3), data=res_)  +
 #geom_ribbon(aes(x=random_rank, ymin = min_test_score, ymax=max_test_score, fill=transform__0__num_features_to_remove), data=res_, alpha=0.3) +
  facet_wrap(~model_name)+
  #geom_text(aes(x=150, y=mean_test_score + 0.0007, color=model_name, label=str_c(model_name, " (Default)")), data=res_ %>% filter(random_rank == 0), size=7) +
  scale_x_log10() +
  xlab("Number of random search iterations") +
  ylab("Normalized test score of best model \n (on valid set) up to this iteration") +
  theme_minimal(base_size=22)+
  theme(legend.position="none")

ggsave("analyses/plots/removed_features.jpg", width=16, height=9)

# plot the difference between score vs difference for gbt

t <- res_ %>%
  select(-max_test_score, -min_test_score) %>% 
  slice_tail(n=1) %>% 
  group_by(model_name) %>% 
  pivot_wider(names_from=transform__0__num_features_to_remove, values_from=mean_test_score) %>% 
  pivot_longer(cols = c("0.01", "0.02", "0.05"), names_to = "threshold", values_to="mean_test_score")


new_tibble <- tibble(gap=(t$"0" - t$mean_test_score) / t$"0", max_rel_decrease=t$threshold, model=t$model_name)


new_tibble %>% 
  mutate(max_rel_decrease = as.numeric(max_rel_decrease)) %>% 
  ggplot() +
  geom_line(aes(x = max_rel_decrease, y = gap, color=model), size=2) +
  geom_line(aes(x=max_rel_decrease, y=max_rel_decrease, size=2)) +
  geom_dl(aes(label = model, x=max_rel_decrease, y=gap, color=model), method = list(dl.combine("smart.grid"), cex=1.3))  +
  xlab("Maximum relative decrease (GBT)") +
  ylab("Actual decrease") +
  theme_minimal(base_size=22)+
  theme(legend.position="none")

ggplot() +
  geom_point(aes(x = transform__0__num_features_to_remove, y = ))


res_ %>%
  select(-max_test_score, -min_test_score) %>% 
  slice_tail(n=1)

# ggplot() +
#   geom_line(aes(x = random_rank, y = mean_test_score, color=model_name, group=model_name), data=res_, size=2) +
#   geom_dl(aes(label = model_name, x=random_rank, y=mean_test_score, color=model_name), method = list(dl.combine("smart.grid"), cex=1.3), data=res_)  +
#   #geom_hline(aes(yintercept=mean_test_score, color=model_name), data=res_ %>% filter(max_rank == 0), linetype="dotted", size=1.5) +
#   #geom_line(aes(x = log_cum_time_factor, y = max_test_score, color=model_name, group=model_name)) +
#   #geom_line(aes(x = log_cum_time_factor, y = min_test_score, color=model_name, group=model_name)) +
#   geom_ribbon(aes(x = random_rank, ymin=min_test_score, ymax=max_test_score, fill=model_name), data=res_, alpha=0.2) +
#   #geom_point(aes(x = cum_time, y = mean_test_score, color=model_name, group=model_name)) +
#   #scale_x_log10(limits= c(10, 1E5)) +
#   scale_x_log10() +
#   #ylim(0.977, 0.993)+
#   xlab("Random search time (seconds)") +
#   ylab("Normalized test score of best model \n (on valid set) up to this time") +
#   theme_minimal(base_size=22) +
#   theme(legend.position="none")


res_time_ <- res_time %>% #select(model_name, data__keyword, log_cum_time_factor,log_cum_time, mean_val_score,best_val_score_so_far, mean_test_score) %>% 
  group_by(model_name, cum_time_factor) %>% 
  summarise(count=sum(!is.na(mean_test_score)),
            max_test_score = mean(mean_test_score, na.rm=T) + 2 * sd(mean_test_score, na.rm=T) / sqrt(count),
            min_test_score = mean(mean_test_score, na.rm=T) - 2 * sd(mean_test_score, na.rm=T) / sqrt(count),
            mean_test_score = mean(mean_test_score, na.rm=T),
            cum_time = mean(cum_time))


library(directlabels)

ggplot() +
  geom_line(aes(x = cum_time, y = mean_test_score, color=model_name, group=model_name), data=res_time_, size=2) +
  geom_dl(aes(label = model_name, x=cum_time, y=mean_test_score, color=model_name), method = list(dl.combine("smart.grid"), cex=1.3), data=res_time_)  +
  #geom_hline(aes(yintercept=mean_test_score, color=model_name), data=res_ %>% filter(max_rank == 0), linetype="dotted", size=1.5) +
  #geom_line(aes(x = log_cum_time_factor, y = max_test_score, color=model_name, group=model_name)) +
  #geom_line(aes(x = log_cum_time_factor, y = min_test_score, color=model_name, group=model_name)) +
  geom_ribbon(aes(x = cum_time, ymin=min_test_score, ymax=max_test_score, fill=model_name), data=res_time_, alpha=0.2) +
  #geom_point(aes(x = cum_time, y = mean_test_score, color=model_name, group=model_name)) +
  scale_x_log10(limits= c(10, 1E5)) +
  #ylim(0.977, 0.993)+
  xlab("Random search time (seconds)") +
  ylab("Normalized test score of best model \n (on valid set) up to this time") +
  theme_minimal(base_size=22) +
  theme(legend.position="none")# +
#scale_x_discrete(breaks=c(10, 60, 3600, 3600*24), labels=c("10 seconds", "1 minute", "1 hour", "1 day"))

ggsave("analyses/plots/random_search_time.jpg", width=16, height=9)


# Iters

#TODO: think per dataset?
res_ <- res %>% 
  group_by(model_name, max_rank) %>% 
  summarise(max_test_score = mean(mean_test_score) + 2 * sd(mean_test_score, na.rm=T) / sqrt(N_SHUFFLE),
            min_test_score = mean(mean_test_score) - 2 * sd(mean_test_score, na.rm=T) / sqrt(N_SHUFFLE),
            mean_test_score = mean(mean_test_score)) 
ggplot() +
  geom_hline(aes(yintercept=mean_test_score, color=model_name), data=res_ %>% filter(max_rank == 0), linetype="dotted", size=1.5) +
  geom_line(aes(x = max_rank, y = mean_test_score, color = model_name), data=res_, size=2) +
  geom_ribbon(aes(x=max_rank, ymin = min_test_score, ymax=max_test_score, fill=model_name), data=res_, alpha=0.3) +
  geom_text(aes(x=150, y=mean_test_score + 0.0004, color=model_name, label=str_c(model_name, " (Default)")), data=res_ %>% filter(max_rank == 0), size=7) +
  scale_x_log10() +
  xlab("Number of random search iterations") +
  ylab("Normalized test score of best model \n (on valid set) up to this iteration") +
  theme_minimal(base_size=22) +
  theme(legend.position="none")







