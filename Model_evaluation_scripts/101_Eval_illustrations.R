# Evaluate model
# Created:    2024-01-16
# Authors:    Christian Vedel [christian-vs@sam.sdu.dk],
#
# Purpose:    Reads predicted labels and produces evaluation results
#
# Output:     paste0("Data/Summary_data/Model_performance", model_name, ".csv")

# ==== Library ====
library(tidyverse)
source("Model_evaluation_scripts/000_Functions.R")
source("Model_evaluation_scripts/001_Generate_eval_stats.R")
source("Model_evaluation_scripts/002_Nature_of_mistakes.R")

# ==== Load eval stats ====
eval_canine = Generate_eval_stats("CANINE", overwrite = FALSE)

# ==== Illustrate some mistakes ====
Nature_of_mistakes("CANINE")

# ==== What is the best threshold? ====
# All
eval_canine %>% filter(summary == "All") %>% 
  plot_of_thresholds("All")

# Langs
x = eval_canine %>% filter(summary == "Lang")
langs = x$lang %>% unique()
for(l in langs) x %>% filter(lang == l) %>% plot_of_thresholds(paste0("Lang_", l))

# ANSWER: Around 0.5 generally

# ==== Performance by language ====
p1 = eval_canine %>% 
  pivot_longer(c(acc, precision, recall, f1), names_to = "stat") %>% 
  filter(summary == "Lang") %>% 
  filter(thr == 0.5) %>% 
  filter(lang_info) %>%
  mutate(
    share_in_training = n/sum(n)
  ) %>% 
  ggplot(aes(share_in_training, value, label = lang)) +
  geom_label(alpha = 0.5, size = 4) +
  facet_wrap(~stat, ncol = 2) + 
  theme_bw() +
  scale_x_continuous(
    labels = scales::percent
  ) +
  scale_y_continuous(
    labels = scales::percent,
    breaks = seq(0.7, 1, by = 0.025),  # Labels at every 0.1
  ) + 
  labs(
    x = "Share of training data",
    y = "Statistic",
    title = "Language_wise performance"
  ) + 
  geom_hline(yintercept = 1)

p1
ggsave("Eval_plots/Performance_lang_wise1.png", width = 10, height = 10, plot = p1)

p1 = eval_canine %>% # Regular bar plot
  pivot_longer(c(acc, precision, recall, f1), names_to = "stat") %>% 
  filter(summary == "Lang") %>% 
  filter(thr == 0.5) %>% 
  filter(lang_info) %>% 
  ggplot(aes(lang, value, fill = n)) +
  geom_bar(stat = "identity") +
  facet_wrap(~stat) + 
  theme_bw() +
  scale_y_continuous(
    labels = scales::percent,
  ) + 
  labs(
    x = "Language",
    y = "Statistic",
    title = "Performance by language"
  ) + 
  geom_hline(yintercept = 1) + 
  theme(
    axis.text.x = element_text(angle = 90, vjust = 0)
  )

p1  
ggsave("Eval_plots/Performance_lang_wise2.png", width = 6, height = 3.5, plot = p1)

# ==== By hisco ====
p1 = eval_canine %>% 
  pivot_longer(c(acc, precision, recall, f1), names_to = "stat") %>% 
  filter(summary == "hisco") %>% 
  filter(thr == 0.5) %>% 
  filter(lang_info) %>% 
  ggplot(aes(n, value, label = hisco_1)) +
  geom_label(alpha = 0.5) +
  facet_wrap(~stat) + 
  theme_bw() +
  scale_x_log10() +
  scale_y_continuous(
    labels = scales::percent,
  ) + 
  labs(
    x = "Threshold",
    y = "Statistic",
    title = "Performance by HISCO code"
  ) + 
  geom_hline(yintercept = 1)

ggsave("Eval_plots/Performance_hisco.png", width = 6, height = 3.5, plot = p1)

p1

  

