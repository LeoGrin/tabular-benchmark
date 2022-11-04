source("analyses/random_search_utils.R")

df <- read_csv("analyses/results/random_search_xps.csv") %>% 
  filter(xp == "add_uninformative_features")
 

res_datasets <- 
  df %>% 
  filter(!is.na(mean_time), !is.na(mean_test_score), !is.na(mean_val_score), !is.na(model_name)) %>% 
  rename() %>% 
  normalize(variable=transform__0__multiplier, normalization_type = "quantile", quantile=0.1) %>% 
  random_search(variable =transform__0__multiplier, n_shuffles=15, default_first = T)


# res_datasets %>% 
#   filter(data__keyword != "poker", data__keyword != "jannis") %>% 
#   filter(random_rank == 20) %>%
#   group_by(model_name, transform__0__multiplier, random_rank, data__keyword) %>% 
#   # mean on shuffles
#   # mean will be NA if at least one score for one dataset is missing for a given time bin
#   summarise(mean_test_score = mean(mean_test_score)) %>% 
#   mutate(transform__0__multiplier = as.numeric(as.character(transform__0__multiplier))) %>% 
#   mutate(percentage_added = as_factor(100 * (transform__0__multiplier - 1))) %>% 
#   mutate(model_name = fct_relevel(model_name, c("RandomForest", "GradientBoostingTree", "FT Transformer"))) %>%
#   ggplot() +
#   geom_boxplot(aes(y = mean_test_score, x =percentage_added, fill=model_name)) +
#   xlab("") +
#   ylab("Normalized test score of best model \n (on valid set) after 10 random search iterations") +
#   labs(fill = "Percentage of \n uninformative \n features added") +
#   theme_minimal(base_size=22)

library(directlabels)
library(ggrepel)
library(shadowtext)
res_datasets_ <- res_datasets %>% 
  # group_by(model_name, data__keyword, n_dataset, transform__0__multiplier) %>% 
  # mutate(num_iters = sum(!is.na(mean_val_score))) %>% 
  # ungroup() %>% 
  # group_by(data__keyword, transform__0__multiplier, n_dataset) %>% 
  # filter(random_rank < min(num_iters)) %>% 
  # ungroup() %>% 
  filter(data__keyword != "poker", data__keyword != "jannis") %>% 
  filter(random_rank == 20) %>%
  group_by(model_name, transform__0__multiplier, random_rank, n_dataset) %>% 
  # mean on shuffles
  # mean will be NA if at least one score for one dataset is missing for a given time bin
  summarise(mean_test_score = mean(mean_test_score)) %>% 
  mutate(transform__0__multiplier = as.numeric(as.character(transform__0__multiplier))) %>% 
  mutate(percentage_added = (transform__0__multiplier - 1)) %>% 
  mutate(model_name = fct_relevel(model_name, c("RandomForest", "GradientBoostingTree", "FT Transformer"))) %>%
  group_by(percentage_added, model_name) %>% 
  summarise(mean_test_score_median = mean(mean_test_score, na.rm=T), 
            mean_test_score_high = quantile(mean_test_score, 0.80, na.rm=T), 
           mean_test_score_low = quantile(mean_test_score, 0.20, na.rm=T))
res_datasets_ %>% 
  ggplot() +
  geom_ribbon(aes(x = percentage_added, ymin = mean_test_score_low, ymax=mean_test_score_high, fill=model_name, group=model_name), alpha=0.3) +
  geom_line(aes(y = mean_test_score_median, x =percentage_added, color=model_name, group=model_name), size=3) +
  geom_point(aes(y = mean_test_score_median, x =percentage_added, color=model_name, group=model_name), size=4) +
  #geom_line(aes(y = mean_test_score, x =percentage_added, color=model_name, group=model_name), size=3) +
  #geom_point(aes(y = mean_test_score, x =percentage_added, color=model_name, group=model_name), size=4) +
  #geom_dl(aes(label = model_name, x=percentage_added, y=mean_test_score_median, color=model_name), 
  #        method = list(dl.combine("smart.grid"), cex=1.6))  +
  #facet_wrap(~data__keyword, scales = "free") +
  geom_text_repel(aes(label=model_name,
                      color = model_name,
                      x = percentage_added,
                      y =  mean_test_score_median),
                  data= (res_datasets_ %>%
                           filter(percentage_added == 0.5)),
                  bg.color='white', size =6.5,bg.r=0.15,
                  nudge_y = 0.02, nudge_x = 0.1
                  , min.segment.length=100)+
  scale_x_continuous(labels = scales::percent) + 
  xlab("Percentage of uninformative \n features added") +
  ylab("Normalized test score of best \n model (on valid set) after \n 30 random search iterations") +
  #labs(fill = "Percentage of \n uninformative \n features added") +
  theme_minimal(base_size=22) + 
  theme(legend.position="none") + 
  colScale


ggsave("analyses/plots/add_features.pdf", width=7, height=6)

##################

# Statistical analysis
library(broom)
tidy(summary(lm(mean_test_score~dataset + model_name + prop_added + model_name * prop_added, 
                data=res_datasets %>% 
                  filter(data__keyword != "poker", data__keyword != "jannis") %>% 
                  filter(random_rank == 20) %>% 
                  mutate(dataset = data__keyword, prop_added = transform__0__multiplier)))) %>% 
  filter(! startsWith(term, "data")) %>% 
  mutate_if(is.numeric, ~round(., 3)) %>% write_csv("analyses/results/tests_add_features.csv")

##################
# Dataset by dataset


res_datasets <- 
  df %>% 
  filter(!is.na(mean_time), !is.na(mean_test_score), !is.na(mean_val_score)) %>% 
  rename() %>% 
  #normalize(variable=transform__0__multiplier, normalization_type = "quantile", quantile=0.1) %>% 
  random_search(variable =transform__0__multiplier, n_shuffles=15, default_first = T)


res_datasets %>% 
  #mutate(model_name = fct_relevel(as_factor(model_name), "GradientBoostingTree", "RandomForest", "FT Transformer", "Resnet"))%>%
  filter(data__keyword != "poker", data__keyword != "jannis") %>% 
  filter(random_rank == 20) %>%
  group_by(model_name, transform__0__multiplier, random_rank, data__keyword) %>% 
  # mean on shuffles
  # mean will be NA if at least one score for one dataset is missing for a given time bin
  summarise(min_test_score = min(mean_test_score),
            max_test_score = max(mean_test_score),
            mean_test_score = mean(mean_test_score)) %>% 
  mutate(transform__0__multiplier = as.numeric(as.character(transform__0__multiplier))) %>% 
  mutate(percentage_added = (transform__0__multiplier - 1)) %>% 
  mutate(model_name = fct_relevel(model_name, c("RandomForest", "GradientBoostingTree", "FT Transformer"))) %>%
  #group_by(percentage_added, model_name) %>% 
  #summarise(mean_test_score_median = mean(mean_test_score, na.rm=T), 
  #          mean_test_score_high = quantile(mean_test_score, 0.80, na.rm=T), 
  #          mean_test_score_low = quantile(mean_test_score, 0.20, na.rm=T)) %>% 
  ggplot() +
  #geom_ribbon(aes(x = percentage_added, ymin = mean_test_score_low, ymax=mean_test_score_high, fill=model_name, group=model_name), alpha=0.3) +
  #geom_line(aes(y = mean_test_score_median, x =percentage_added, color=model_name, group=model_name), size=3) +
  #geom_point(aes(y = mean_test_score_median, x =percentage_added, color=model_name, group=model_name), size=4) +
  geom_line(aes(y = mean_test_score, x =percentage_added, color=model_name, group=model_name), size=2) +
  geom_point(aes(y = mean_test_score, x =percentage_added, color=model_name, group=model_name), size=3) +
  geom_ribbon(aes(y = mean_test_score, x =percentage_added, ymax=max_test_score, ymin=min_test_score, fill=model_name, group=model_name), alpha=0.3) +
  #geom_dl(aes(label = model_name, x=percentage_added, y=mean_test_score_median, color=model_name), 
  #        method = list(dl.combine("smart.grid"), cex=1.6))  +
  facet_wrap(~data__keyword, scales = "free") +
  scale_x_continuous(labels = scales::percent) + 
  xlab("Percentage of uninformative features added") +
  ylab("Test accuracy of best \n model (on valid set) after \n 20 random search iterations") +
  labs(fill = "Percentage of \n uninformative \n features added") +
  theme_minimal(base_size=22) + 
  #theme(legend.position="none") + 
  theme(legend.position="bottom", legend.title=element_blank(), legend.text = element_text(size=22)) +
  colScale

ggsave("analyses/plots/add_features_datasets.pdf", width=15, height=10)
 

