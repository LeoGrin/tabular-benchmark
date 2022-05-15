library(tidyverse)

df <- tibble()

for (N in c(20, 50, 100, 1000, 1E4, 1E5)) {
  samples_J = rbeta(1000, 1+5, 1+N-5)
  samples_T = rbeta(1000, 1+10, 1+N-10)
  samples_JnT = rbeta(1000, 1+5, 1+N-5)
  df_new <- tibble(p_indep = samples_J*samples_T, p_real = samples_JnT)
  df <- df %>% bind_rows(df_new %>% mutate(n_samples = N))
}

df %>% 
  ggplot() +
  geom_histogram(aes(x=p_indep, fill="Probabilty of (Jailed and Tall) if Jailed and Tall were independent"), alpha=0.8) +
  geom_histogram(aes(x=p_real, fill="Probability of (Jailed and Tall) observered in the data"), alpha=0.8) +
  facet_wrap(~n_samples, scales = "free") +
  ggtitle("Distribution of the probability of (Jailed and Tall) either \n estimating P(Jailed) and P(Tall) from the data and assuming \n independence, or estimating P(Jailed and Tall) directly from the data ")



df <- tibble(x = rbinom(1000, 1E5, 5 / 1E5))

df %>% 
  ggplot() +
  geom_histogram(aes(x = x))
