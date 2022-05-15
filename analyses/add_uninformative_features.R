source("analyses/random_search_utils.R")

#old results
#df <- read_csv("results/sweeps/add_features/resnet_add_features.csv") %>% 
#  bind_rows(read_csv("results/sweeps/add_features/ft_transformer_add_features.csv")) %>% 
#  bind_rows(read_csv("results/sweeps/add_features/rf_add_features.csv")) %>% 
#  bind_rows(read_csv("results/sweeps/add_features/gpt_add_features.csv")) %>% 
##  mutate(transform__0__multiplier = as_factor(transform__0__multiplier)) %>% 
#  filter(data__keyword %in% c("heloc", "electricity", "california", "covtype", "churn", "cpu", "wine"))


df <- read_csv("results/sweeps/sweeps_classif/add_useless/ft_transformer_add_features.csv") %>% 
  bind_rows(read_csv("results/sweeps/sweeps_classif/add_useless/rf_add_features.csv")) %>% 
  bind_rows(read_csv("results/sweeps/sweeps_classif/add_useless/gbt_add_features.csv")) %>% 
  bind_rows(read_csv("results/sweeps/sweeps_classif/add_useless/resnet_add_features.csv"))


res_datasets <- 
  df %>% 
  filter(!is.na(mean_time)) %>% 
  rename() %>% 
  normalize(variable=transform__0__multiplier) %>% 
  random_search(variable =transform__0__multiplier, n_shuffles=20)


library(RColorBrewer)
gg_color_hue <- function(n) {
  hues = seq(15, 375, length = n + 1)
  hcl(h = hues, l = 65, c = 100)[1:n]
}
myColors <- gg_color_hue(6)
names(myColors) <- c("GradientBoostingTree", "RandomForest", "XGBoost", "FT Transformer", "Resnet", "MLP")
colScale <- list(scale_colour_manual(name = "grp",values = myColors),
                 scale_fill_manual(name = "grp",values = myColors))

res_datasets %>% 
  filter(data__keyword != "poker", data__keyword != "jannis") %>% 
  filter(random_rank == 20) %>%
  group_by(model_name, transform__0__multiplier, random_rank, data__keyword) %>% 
  # mean on shuffles
  # mean will be NA if at least one score for one dataset is missing for a given time bin
  summarise(mean_test_score = mean(mean_test_score)) %>% 
  mutate(transform__0__multiplier = as.numeric(as.character(transform__0__multiplier))) %>% 
  mutate(percentage_added = as_factor(100 * (transform__0__multiplier - 1))) %>% 
  mutate(model_name = fct_relevel(model_name, c("RandomForest", "GradientBoostingTree", "FT Transformer"))) %>%
  ggplot() +
  geom_boxplot(aes(y = mean_test_score, x =percentage_added, fill=model_name)) +
  xlab("") +
  ylab("Normalized test score of best model \n (on valid set) after 10 random search iterations") +
  labs(fill = "Percentage of \n uninformative \n features added") +
  theme_minimal(base_size=22)


res_datasets %>% 
  filter(data__keyword != "poker", data__keyword != "jannis") %>% 
  filter(random_rank == 20) %>%
  group_by(model_name, transform__0__multiplier, random_rank, data__keyword) %>% 
  # mean on shuffles
  # mean will be NA if at least one score for one dataset is missing for a given time bin
  summarise(mean_test_score = mean(mean_test_score)) %>% 
  mutate(transform__0__multiplier = as.numeric(as.character(transform__0__multiplier))) %>% 
  mutate(percentage_added = (transform__0__multiplier - 1)) %>% 
  mutate(model_name = fct_relevel(model_name, c("RandomForest", "GradientBoostingTree", "FT Transformer"))) %>%
  group_by(percentage_added, model_name) %>% 
  summarise(mean_test_score_median = mean(mean_test_score, na.rm=T), 
            mean_test_score_high = quantile(mean_test_score, 0.80, na.rm=T), 
            mean_test_score_low = quantile(mean_test_score, 0.20, na.rm=T)) %>% 
  ggplot() +
  geom_ribbon(aes(x = percentage_added, ymin = mean_test_score_low, ymax=mean_test_score_high, fill=model_name, group=model_name), alpha=0.3) +
  geom_line(aes(y = mean_test_score_median, x =percentage_added, color=model_name, group=model_name), size=3) +
  geom_point(aes(y = mean_test_score_median, x =percentage_added, color=model_name, group=model_name), size=4) +
  geom_dl(aes(label = model_name, x=percentage_added, y=mean_test_score_median, color=model_name), method = list(dl.combine("smart.grid"), cex=1.6))  +
  scale_x_continuous(labels = scales::percent) + 
  xlab("Percentage of uninformative features added") +
  ylab("Normalized test score of best model \n (on valid set) after 10 random search iterations") +
  labs(fill = "Percentage of \n uninformative \n features added") +
  theme_minimal(base_size=22) + 
  theme(legend.position="none") + 
  colScale
  

ggsave("analyses/plots/add_features.jpg", width=16, height=9)






