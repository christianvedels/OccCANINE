# Evaluate model
# Created:    2024-01-16
# Authors:    Christian Vedel [christian-vs@sam.sdu.dk],
#
# Purpose:    Reads predicted labels and produces evaluation results
#
# Output:     paste0("Data/Summary_data/Model_performance", model_name, ".csv")

# ==== Params ==== 
dim = c(6, 9)

# ==== Library ====
library(tidyverse)
library(knitr)
library(kableExtra)
library(hisco)
library(fixest)
library(fwildclusterboot)

source("Model_evaluation_scripts/000_Functions.R")
source("Model_evaluation_scripts/001_Generate_eval_stats.R")
source("Model_evaluation_scripts/002_Nature_of_mistakes.R")

# ==== Load eval stats ====
eval_canine = Generate_eval_stats("CANINE", overwrite = FALSE, toyrun = FALSE)

# ==== Illustrate some mistakes ====
# Nature_of_mistakes("CANINE")

# ==== What is the best threshold? ====
# All
p1 = eval_canine %>% filter(summary == "All") %>% 
  mutate(
    lang_info = ifelse(lang_info, "With language info.", "Without language info.")
  ) %>% 
  plot_of_thresholds("All") + 
  theme(legend.position = "bottom") +
  labs(title = NULL, subtitle = NULL) + 
  scale_color_manual(values = c(red, green)) + 
  labs(
    col = "",
    shape = "",
    y = "Statistic"
  )

p1
ggsave("Project_dissemination/Figures for paper/Threshold_and_performance.png", plot = p1, height = dim[1], width = dim[2], dpi = 600)
ggsave("Project_dissemination/Figures for paper/Threshold_and_performance.pdf", plot = p1, height = dim[1], width = dim[2])

# Best overall thresholds
all_stats = eval_canine %>% 
  filter(summary == "All") %>% 
  pivot_longer(c(acc, precision, recall, f1), names_to = "stat") %>% 
  group_by(stat, lang_info) %>% 
  filter(value == max(value))

the_best = all_stats %>% 
  select(stat, thr, lang_info, value)

# ==== Table for paper of best overall performance/threshold ====
the_best %>%
  arrange(stat, lang_info) %>% 
  select(stat, lang_info, value, thr) %>%
  mutate(
    stat = case_when(
      stat == "acc" ~ "Accuracy",
      stat == "f1" ~ "F1 score",
      stat == "precision" ~ "Precision",
      stat == "recall" ~ "Recall"
    )
  ) %>% 
  mutate(
    lang_info = ifelse(lang_info, "Yes", "No")
  ) %>% 
  rename(
    `Optimal thr.` = thr,
    Statistic = stat,
    Value = value,
    `Lang. info.` = lang_info
  ) %>% 
  mutate(
    Value = signif(Value, 3)
  ) %>% 
  kable("latex", booktabs = TRUE, caption = "Your Caption Here") %>% 
  collapse_rows(columns = c(1), valign = "middle")


# ==== Performance by lang ====
# Langs
x = eval_canine %>% filter(summary == "Lang")
n_all = eval_canine %>% filter(summary == "All") %>% distinct(n) %>% unlist()
langs = x$lang %>% unique()
best_lang = foreach(l = langs, .combine = "bind_rows") %do% {
  x_l = x %>% 
    filter(lang == l)
  
  max_f1 = x_l %>% 
    filter(f1 == max(f1))
  
  max_acc = x_l %>% 
    filter(acc == max(acc))
  
  max_prec = x_l %>% 
    filter(precision == max(precision))
  
  max_recall = x_l %>% 
    filter(recall == max(recall))
  
  x_l %>% 
    plot_of_thresholds(paste0("Lang_", l))
  
  return(
    data.frame(
      lang = l,
      thr = c(max_acc$thr[1], max_f1$thr[1], max_prec$thr[1], max_recall$thr[1]),
      stat = c("acc", "f1", "precision", "recall"),
      value = c(max_acc$acc[1], max_f1$f1[1], max_prec$precision[1], max_recall$recall[1]),
      n = max_f1$n[1],
      pct_of_train = max_f1$n[1]/n_all
    )
  )
}

all_stats = eval_canine %>% 
  filter(summary == "All") %>% 
  pivot_longer(c(acc, precision, recall, f1), names_to = "stat") %>% 
  group_by(stat) %>% 
  filter(value == max(value)) %>% 
  mutate(
    lang = "it"
  ) %>% 
  mutate(
    stat = case_when(
      stat == "acc" ~ "Accuracy",
      stat == "f1" ~ "F1 score",
      stat == "precision" ~ "Precision",
      stat == "recall" ~ "Recall"
    )
  )

plot_data = best_lang %>%  
  filter(lang != "ge") %>% # Mistake in training data
  group_by(stat) %>% 
  mutate(
    lang = factor(lang, levels = lang[order(-value)])
  ) %>% 
  mutate(
    label = paste0(scales::percent(value, 0.1), " (", thr, ")" )
  )

p1 = plot_data %>% 
  mutate(
    stat = case_when(
      stat == "acc" ~ "Accuracy",
      stat == "f1" ~ "F1 score",
      stat == "precision" ~ "Precision",
      stat == "recall" ~ "Recall"
    )
  ) %>% 
  ggplot(aes(lang, value)) + 
  geom_bar(stat = "identity", alpha = 0.8, fill = red) +
  scale_y_continuous(
    labels = scales::percent,
    breaks = seq(0,1, by = 0.1)
  ) + 
  theme_bw() + 
  facet_wrap(~stat, scales = "free") + 
  geom_text(
    y = 0.4,
    aes(label = label, x = lang),
    col = "grey", 
    angle = 90
  ) + 
  geom_hline(aes(yintercept = value), data = all_stats, lty = 2) + 
  geom_text(
    data = all_stats,
    aes(label = scales::percent(value, 0.1), x = lang, y = value-0.075),
    inherit.aes = FALSE
  ) + 
  theme(
    axis.text.x = element_text(angle = 90, vjust = 0.5)
  ) + 
  labs(
    y = "Statistic",
    x = "Language"
  )
  
p1
ggsave("Project_dissemination/Figures for paper/Performance_by_language.png", plot = p1, height = dim[1], width = dim[2], dpi = 600)
ggsave("Project_dissemination/Figures for paper/Performance_by_language.pdf", plot = p1, height = dim[1], width = dim[2])

# Same plot but only F1
p1 = plot_data %>% 
  filter(stat == "f1") %>% 
  mutate(lang = factor(as.character(lang), levels = lang[order(-value)])) %>% 
  ggplot(aes(lang, value)) + 
  geom_bar(stat = "identity", alpha = 0.8, fill = red) +
  scale_y_continuous(
    labels = scales::percent,
    breaks = seq(0,1, by = 0.1)
  ) + 
  theme_bw() + 
  geom_text(
    y = 0.4,
    aes(label = label, x = lang),
    col = "grey", 
    angle = 90,
    size = 6
  ) + 
  geom_hline(
    aes(yintercept = 0.9), 
    data = all_stats %>% filter(stat == "f1"), 
    lty = 2
  ) + 
  geom_text(
    data = all_stats %>% filter(stat == "f1"),
    aes(label = "90%", x = "is", y = 0.95),
    inherit.aes = FALSE,
    size = 8
  ) + 
  theme(
    axis.text.x = element_text(angle = 90, vjust = 0.5)
  ) + 
  labs(x = "Language", y = "F1")
p1
ggsave("Project_dissemination/Youtube_video/Slides/Figures/F1_multiling.png", plot = p1, height = dim[1], width = dim[2], dpi = 600)

# ==== Table of optimal thresholds for appendix ====
plot_data %>%
  select(lang, n, stat, value, thr) %>%
  mutate(
    stat = case_when(
      stat == "acc" ~ "Accuracy",
      stat == "f1" ~ "F1 score",
      stat == "precision" ~ "Precision",
      stat == "recall" ~ "Recall"
    )
  ) %>% 
  rename(
    Language = lang,
    `Optimal thr.` = thr,
    Statistic = stat,
    Value = value,
    `N. test obs.` = n
  ) %>% 
  kable("latex", booktabs = TRUE, longtable = TRUE, caption = "Your Caption Here") %>%
  kable_styling(latex_options = c("repeat_header"), repeat_header_text = "Continued: ") %>% 
  collapse_rows(columns = c(1, 2), valign = "middle")

# ==== Performance by language ====
all_stats = eval_canine %>% 
  filter(summary == "All") %>% 
  pivot_longer(c(acc, precision, recall, f1), names_to = "stat") %>% 
  group_by(stat) %>% 
  filter(value == max(value))



p1 = eval_canine %>% 
  pivot_longer(c(acc, precision, recall, f1), names_to = "stat") %>% 
  filter(summary == "Lang") %>% 
  filter(thr == 0.5) %>% 
  filter(lang_info) %>%
  mutate(
    share_in_training = n/sum(n)
  ) %>% 
  mutate(
    stat = case_when(
      stat == "acc" ~ "Accuracy",
      stat == "f1" ~ "F1 score",
      stat == "precision" ~ "Precision",
      stat == "recall" ~ "Recall"
    )
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
ggsave("Eval_plots/Performance_lang_wise1.png", width = 10, height = 10, plot = p1, dpi = 600)
ggsave("Eval_plots/Performance_lang_wise1.pdf", width = 10, height = 10, plot = p1, dpi = 600)

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
ggsave("Eval_plots/Performance_lang_wise2.png", width = 6, height = 3.5, plot = p1, dpi = 600)
ggsave("Eval_plots/Performance_lang_wise2.pdf", width = 6, height = 3.5, plot = p1, dpi = 600)

# ==== By hisco ====
the_best0 = the_best %>% filter(lang_info)

plot_data = eval_canine %>% 
  pivot_longer(c(acc, precision, recall, f1), names_to = "stat") %>% 
  left_join(the_best0, by = "stat", suffix = c("","_best")) %>% 
  filter("hisco" == summary) %>% 
  filter(thr == thr_best) %>% 
  filter(lang_info) %>% 
  mutate(
    pct_of_train = n/all_stats$n[1]
  ) %>% 
  group_by(stat) %>% 
  arrange(-pct_of_train) %>% 
  mutate(pct_cum_sum = cumsum(pct_of_train)) %>%
  mutate(
    stat = case_when(
      stat == "acc" ~ "Accuracy",
      stat == "f1" ~ "F1 score",
      stat == "precision" ~ "Precision",
      stat == "recall" ~ "Recall"
    )
  )

q99 = plot_data %>% 
  group_by(stat) %>% 
  filter(pct_cum_sum > 0.99) %>% 
  filter(pct_cum_sum == min(pct_cum_sum))

labels99 = plot_data %>% 
  group_by(stat) %>% 
  summarise(
    n_label = quantile(n*14, p = 0.95)
  ) %>% 
  mutate(
    y = 0.4,
    label = "99% of\nobservations"
  )

labels01 = plot_data %>% 
  group_by(stat) %>% 
  summarise(
    n_label = quantile(n*14, p = 0.20)
  ) %>% 
  mutate(
    y = 0.4,
    label = "1% of\nobservations"
  )

all_stats = all_stats %>%
  mutate(
    stat = case_when(
      stat == "acc" ~ "Accuracy",
      stat == "f1" ~ "F1 score",
      stat == "precision" ~ "Precision",
      stat == "recall" ~ "Recall"
    )
  )
  

p1 = plot_data %>% 
  mutate(
    n_approx = n*14
  ) %>% 
  drop_na(stat) %>%
  ggplot(aes(n_approx, value, label = hisco_1)) +
  geom_smooth(se = FALSE, col = red) + 
  geom_point(size = 0.1, shape = 4) + 
  facet_wrap(~stat) + 
  theme_bw() +
  scale_y_continuous(
    labels = scales::percent,
  ) + 
  scale_x_log10() +
  labs(
    x = "N in training",
    y = "Statistic"
  ) + 
  geom_hline(yintercept = 1) + 
  geom_hline(aes(yintercept = value), data = all_stats, lty = 3) +
  geom_vline(aes(xintercept = n*14), data = q99, lty = 3) +
  theme(
    axis.text.x = element_text(angle = 90, vjust = 0.5)
  ) + 
  geom_text(
    aes(x = n_label, y = y, label = label), data = labels99
  ) + 
  geom_text(
    aes(x = n_label, y = y, label = label), data = labels01
  )

p1
ggsave("Eval_plots/Performance_hisco.png", width = 6, height = 3.5, plot = p1)
ggsave("Project_dissemination/Figures for paper/Performance_by_hisco.png", plot = p1, height = dim[1], width = dim[2], dpi = 600)
ggsave("Project_dissemination/Figures for paper/Performance_by_hisco.pdf", plot = p1, height = dim[1], width = dim[2])

p1

# N HISCO codes above / below 
cutoff = q99$n %>% unique()

plot_data %>% 
  group_by(stat) %>% 
  filter(n >= cutoff) %>%
  count()

# ==== Performance by SES ====
ses_data = eval_canine %>% 
  filter(lang_info) %>% 
  pivot_longer(c(acc, precision, recall, f1), names_to = "stat") %>% 
  left_join(the_best0, by = "stat", suffix = c("","_best")) %>% 
  filter("hisco" == summary) %>% 
  filter(thr == thr_best) %>% 
  filter(summary == "hisco") %>% 
  mutate(
    ses_value = hisco_to_ses(hisco_1, ses = "hiscam_u1")
  ) %>% 
  mutate(
    stat = case_when(
      stat == "acc" ~ "Accuracy",
      stat == "f1" ~ "F1 score",
      stat == "precision" ~ "Precision",
      stat == "recall" ~ "Recall"
    )
  )


p1 = ses_data %>% 
  ggplot(aes(ses_value, value)) + 
  facet_wrap(~stat) + 
  geom_point(size = 0.1, shape = 4)  + 
  geom_smooth(col = red) +
  theme_bw() +
  labs(y = "Statistic", x = "Socio-Economic Score (HISCAM)")

p1
ggsave("Project_dissemination/Figures for paper/Performance_by_ses.png", plot = p1, height = dim[1], width = dim[2], dpi = 600)
ggsave("Project_dissemination/Figures for paper/Performance_by_ses.pdf", plot = p1, height = dim[1], width = dim[2])

# Regressions to test ses_value ~ model performance
regdata = ses_data %>%
  pivot_wider(names_from = stat, values_from = value, id_cols = c(hisco_1, ses_value, n)) %>%
  mutate(
    n = n*14 # To scale to the amount of training data used
  ) %>% 
  rename(
    F1score = `F1 score`
  ) %>% 
  drop_na()

mod1_acc = feols(
  Accuracy ~ ses_value, 
  data = regdata, 
  vcov = "hetero"
)
mod2_acc = feols(
  Accuracy ~ ses_value + log(n), 
  data = regdata, 
  vcov = "hetero"
)
mod1_f1 = feols(
  F1score~ses_value, 
  data = regdata, 
  vcov = "hetero"
)
mod2_f1 = feols(
  F1score~ses_value + log(n), 
  data = regdata, 
  vcov = "hetero"
)

mod1_prec = feols(
  Precision~ses_value, 
  data = regdata, 
  vcov = "hetero"
)
mod2_prec = feols(
  Precision~ses_value + log(n), 
  data = regdata, 
  vcov = "hetero"
)

mod1_recall = feols(
  Recall~ses_value, 
  data = regdata, 
  vcov = "hetero"
)
mod2_recall = feols(
  Recall~ses_value + log(n), 
  data = regdata, 
  vcov = "hetero"
)

etable(
  mod1_acc, mod1_f1, mod1_prec, mod1_recall,
  drop = "Constant",
  tex = TRUE,
  coefstat = "confint",
  fitstat = "n",
  signif.code=NA
)

etable(
  mod2_acc, mod2_f1, mod2_prec, mod2_recall,
  drop = "Constant",
  tex = TRUE,
  coefstat = "confint",
  fitstat = "n",
  signif.code=NA
)

# ==== Cohens kappa ====
the_best

x = eval_canine %>% 
  filter(summary == "All") %>% 
  filter(lang_info) 

x %>% 
  ggplot(aes(thr, c_kappa)) + 
  geom_point() +
  theme_bw()

x %>% 
  filter(c_kappa == max(c_kappa))
