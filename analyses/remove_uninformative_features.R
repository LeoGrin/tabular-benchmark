library(tidyverse)


df <- read_csv("analyses/results/random_search_xps.csv") %>% 
  filter(xp == "remove_uninformative_features")



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
