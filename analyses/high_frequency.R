source("analyses/random_search_utils.R")
library(directlabels)
library(ggrepel)
library(shadowtext)


df <- read_csv("analyses/results/random_search_xps.csv") %>% 
  filter(xp == "high_frequencies")

res_datasets <- 
df %>% 
#filter(transform__2__cov_mult %in% c(0, 0.001, 0.01, 0.1, 0.5)) %>% 
filter(!is.na(mean_time)) %>% 
rename() %>% 
normalize(variable=transform__2__cov_mult, normalization_type = "quantile", quantile = 0.1) %>% 
random_search(variable =transform__2__cov_mult, n_shuffles=15, default_first = T)
  
# 
# res_datasets %>% 
#   filter(random_rank == 100) %>%
#   filter(model_name != "MLP") %>%
#   filter(data__keyword != "poker", data__keyword != "jannis") %>% #, data__keyword %in% c("electricity", "california")) %>% 
#   #filter(data__keyword == "wine") %>% 
#   group_by(model_name, data__keyword, transform__2__cov_mult) %>% 
#   summarise(mean_test_score = mean(mean_test_score)) %>% 
#   mutate(transform__2__cov_mult = as_factor(transform__2__cov_mult)) %>% 
#   ungroup() %>% 
#   pivot_wider(names_from=transform__2__cov_mult, values_from=mean_test_score) %>% 
#   mutate(diff = `0` - `0.1`) %>% 
#   ggplot() +
#   #geom_jitter(aes(x = model_name, y=diff, color=data__keyword), width=0.1, height=0)
#   geom_text(aes(x = model_name, y=diff, label=data__keyword))
#   #geom_boxplot(aes(x = model_name, y=diff, fill=model_name))
# 
# 
# library(RColorBrewer)
# gg_color_hue <- function(n) {
#   hues = seq(15, 375, length = n + 1)
#   hcl(h = hues, l = 65, c = 100)[1:n]
# }
# myColors <- gg_color_hue(6)
# names(myColors) <- c("GradientBoostingTree", "RandomForest", "XGBoost", "FT Transformer", "Resnet", "MLP")
# colScale <- list(scale_colour_manual(name = "grp",values = myColors),
#                  scale_fill_manual(name = "grp",values = myColors))
# 
# 




res_datasets_ <- res_datasets %>% 
  filter(random_rank == 60) %>%
  filter(model_name != "MLP") %>%
  filter(data__keyword != "poker") %>% 
  group_by(model_name, n_dataset, transform__2__cov_mult) %>% 
  summarise(mean_test_score = mean(mean_test_score)) %>% 
  filter(transform__2__cov_mult < 0.5) %>% 
  mutate(transform__2__cov_mult = as_factor(transform__2__cov_mult)) %>% 
  ungroup() %>% 
  mutate(model_name = fct_relevel(as_factor(model_name), "GradientBoostingTree", "RandomForest", "FT Transformer", "Resnet"))
  # pivot_wider(names_from=transform__1__cov_mult, values_from=mean_test_score) %>% 

res_datasets_2 <- res_datasets_ %>% mutate(transform__2__cov_mult_num = as.numeric(transform__2__cov_mult)) %>% 
  group_by(model_name, n_dataset) %>% 
  mutate(r = rank(transform__2__cov_mult_num), background_color = if_else(r %%2 == 0, 0.01, 0.0)) %>% 
  ungroup()

small_df <- res_datasets_2 %>% filter(n_dataset==1, model_name == "FT Transformer")
  
res_datasets_2 %>% 
  ggplot() +
  geom_rect(data = small_df, aes(ymin=-Inf, ymax=Inf, xmin=transform__2__cov_mult_num - 0.5, xmax=transform__2__cov_mult_num + 0.5, alpha=background_color))+
  geom_boxplot(aes(x = transform__2__cov_mult_num, y=mean_test_score, fill=model_name, group=interaction(transform__2__cov_mult_num, model_name))) + #, width=0.1, height=0) +
  geom_text_repel(aes(label=model_name,
                      color = model_name,
                      x = transform__2__cov_mult_num,
                      y =  mean_test_score),
                  data= (res_datasets_2 %>%
                           filter(transform__2__cov_mult == "0", n_dataset==1)),
                  bg.color='white', size =6.5,bg.r=0.15,
                  nudge_y = -0.01, nudge_x = 0.7
                  , min.segment.length=1000)+
  scale_x_continuous(breaks=small_df$transform__2__cov_mult_num, labels=small_df$transform__2__cov_mult) + 
  scale_alpha(range = c(0, 0.13)) +  
  xlab("(Squared) lengthscale of the \n Gaussian kernel smoother") + 
  ylab("Normalized test score of best model \n (on valid set) after 60 random search \n iterations, averaged across datasets") +
  theme_minimal(base_size=22) +
  theme(legend.position="none")+
  colScale



ggsave("analyses/plots/high_frequencies.pdf", width=7, height=6.1)

##########
# Statistical analysis
library(broom)
tidy(summary(lm(mean_test_score~dataset + model_name + lengthscale + model_name * lengthscale, 
                data=res_datasets %>% 
                  filter(random_rank == 60) %>%
                  filter(model_name != "MLP") %>%
                  filter(data__keyword != "poker") %>% 
                  mutate(dataset = data__keyword, lengthscale = transform__2__cov_mult)))) %>% 
  filter(! startsWith(term, "data")) %>% 
           mutate_if(is.numeric, ~round(., 3)) %>% write_csv("analyses/results/tests_hf.csv")

#################@
# Dataset by dataset

res_datasets <- 
  df %>% 
  #filter(transform__2__cov_mult %in% c(0, 0.001, 0.01, 0.1, 0.5)) %>% 
  filter(!is.na(mean_time)) %>% 
  rename() %>% 
  #normalize(variable=transform__2__cov_mult, normalization_type = "quantile", quantile = 0.1) %>% 
  random_search(variable =transform__2__cov_mult, n_shuffles=15, default_first = T)




res_datasets_ <- res_datasets %>% 
  filter(random_rank == 60) %>%
  filter(model_name != "MLP") %>%
  filter(data__keyword != "poker") %>% 
  #group_by(model_name, n_dataset, transform__2__cov_mult) %>% 
  #summarise(mean_test_score = mean(mean_test_score)) %>% 
  filter(transform__2__cov_mult < 0.5) %>% 
  mutate(transform__2__cov_mult = as_factor(transform__2__cov_mult)) %>% 
  ungroup() %>% 
  mutate(model_name = fct_relevel(as_factor(model_name), "GradientBoostingTree", "RandomForest", "FT Transformer", "Resnet"))
# pivot_wider(names_from=transform__1__cov_mult, values_from=mean_test_score) %>% 

res_datasets_2 <- res_datasets_ %>% mutate(transform__2__cov_mult_num = as.numeric(transform__2__cov_mult)) %>% 
  group_by(model_name, n_dataset) %>% 
  mutate(r = rank(transform__2__cov_mult_num), background_color = if_else(r %%2 == 0, 0.01, 0.0)) %>% 
  ungroup()

small_df <- res_datasets_2 %>% filter(n_dataset==1, model_name == "FT Transformer")

small_df_2 <- res_datasets_2 %>% filter(n_dataset==1, model_name == "FT Transformer", data__keyword == "electricity") # prevent axis text aliasing

res_datasets_2 %>% 
  ggplot() +
  geom_rect(data = small_df, aes(ymin=-Inf, ymax=Inf, xmin=transform__2__cov_mult_num - 0.5, xmax=transform__2__cov_mult_num + 0.5, alpha=background_color))+
  geom_boxplot(aes(x = transform__2__cov_mult_num, y=mean_test_score, color=model_name, group=interaction(transform__2__cov_mult_num, model_name))) + #, width=0.1, height=0) +
  facet_wrap(~data__keyword, scales="free") +
  scale_x_continuous(breaks=small_df_2$transform__2__cov_mult_num, labels=small_df_2$transform__2__cov_mult) + 
  scale_alpha(range = c(0, 0.13)) +  
  xlab("(Squared) lengthscale of the \n Gaussian kernel smoother") + 
  ylab("Test accuracy of best model \n (on valid set) after 60 random search \n iterations, averaged across datasets") +
  theme_minimal(base_size=22) +
  theme(legend.position="bottom", legend.title=element_blank(), legend.text = element_text(size=22)) +
  colScale +
  guides(alpha="none")

ggsave("analyses/plots/high_frequencies_datasets.pdf", width=15, height=10)






