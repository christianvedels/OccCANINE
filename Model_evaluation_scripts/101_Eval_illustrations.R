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

# ==== Load eval stats ====
eval_canine = Generate_eval_stats("CANINE", overwrite = FALSE)

# ==== What is the best threshold? ====
plot_of_thresholds = function(x, name){
  
  plot_stats = x %>% 
    pivot_longer(c(acc, precision, recall, f1), names_to = "stat") %>% 
    group_by(stat, lang_info) %>% 
    mutate(
      best = max(value) == value
    ) %>% 
    # filter(!lang_info) %>%  # Something weird happens when lang is included
    ungroup()
  
  p1 = plot_stats %>% 
    ggplot(aes(thr, value, col = lang_info)) + 
    geom_point() + 
    geom_line(lty = 2) +
    geom_point(data = subset(plot_stats, best), aes(thr, value), shape = 4, size = 3, col = "black") +  # Highlight max points
    geom_vline(data = subset(plot_stats, best), aes(xintercept = thr), lty = 2) +
    geom_hline(data = subset(plot_stats, best), aes(yintercept = value), lty = 2) +
    facet_wrap(~stat) +  # Changed from ~name to ~stat based on your data structure
    theme_bw() +
    scale_y_continuous(
      labels = scales::percent,
      breaks = seq(floor0(min(plot_stats$value),1), 1, by = 0.025),  # Labels at every 0.1
    ) + 
    scale_x_continuous(
      breaks = seq(0, 1, by = 0.1),  # Labels at every 0.1
      minor_breaks = seq(0, 0, by = 0.05) # Lines at every 0.05
    ) + 
    labs(
      x = "Threshold",
      y = "Statistic",
      title = name,
      subtitle = paste("N =", plot_stats$n[1])
    )
  
  path = paste0("Eval_plots/Optimal_threshold/", name, ".png")
  ggsave(path, width = 10, height = 8, plot = p1)
  
  return(p1)
}

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
    x = "Threshold",
    y = "Statistic",
    title = "Performance by language"
  ) + 
  geom_hline(yintercept = 1)

p1  
ggsave("Eval_plots/Performance_lang_wise2.png", width = 10, height = 8, plot = p1)

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

ggsave("Eval_plots/Performance_hisco.png", width = 10, height = 8, plot = p1)

p1

  

