library(tidyverse)

df <- read_csv("predictions/rf_predictions_wine_train.csv") %>% 
  left_join(read_csv("predictions/mlp_predictions_california_train.csv"), c("x_0", "x_1", "y_true"), suffix = c(".rf", ".mlp"))
#df <- read_csv("mlp_predictions_elec_train.csv")
df <- read_csv("predictions/mlp_predictions_california_train.csv")

df %>% 
  mutate(y_true=as_factor(y_true)) %>% 
  #filter(y_pred.mlp != y_pred.rf) %>% 
  ggplot() + 
  geom_point(aes(x=x_0, y=x_1, color=y_pred,shape=y_true), alpha=0.4, size=1) +
  xlim(-3, 3) +
  ylim(-3, 3)

df %>% 
  mutate(y_true=as_factor(y_true)) %>% 
  #filter(y_pred.mlp != y_pred.rf) %>% 
  ggplot() + 
  geom_point(aes(x=x_0, y=x_1, color=y_pred.mlp == y_true,shape=y_true), alpha=0.4, size=1) +
  xlim(-3, 3) +
  ylim(-3, 3)

df %>% 
  mutate(y_true=as_factor(y_true)) %>% 
  #filter(y_pred.mlp != y_pred.rf) %>% 
  ggplot() + 
  geom_point(aes(x=x_0, y=x_1, color=y_pred.rf == y_true & y_pred.mlp != y_true,shape=y_true), alpha=0.4, size=1) +
  xlim(-3, 3) +
  ylim(-3, 3)

df %>% 
  mutate(y_true=as_factor(y_true)) %>% 
  #filter(y_pred.mlp != y_pred.rf) %>% 
  ggplot() + 
  geom_point(aes(x=x_0, y=x_1, color=y_true, shape=y_true), alpha=0.4, size=1) +
  xlim(-3, 3) +
  ylim(-3, 3)


df %>% 
  select(x_1) %>% 
  distinct()
