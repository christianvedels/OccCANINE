# Perforamance by frequency and SES
# Author: Christian Vedel [christian-vs@sam.sdu.dk]
# Date: 2025-07-12
#
# Purpose: Evaluate model performance by frequency of occurrence and socio-economic status (SES)

# ==== Libraries ====
library(tidyverse)
library(hisco)
library(fixest)
source("Data_cleaning_scripts/000_Functions.R")

# ==== Load data ====
# Can be produced from the predict_test.py script
# Data available on request

flat = read_csv("Project_dissemination/Paper_replication_package/Data/test_performance/obs_test_performance_flat.csv")
greedy = read_csv("Project_dissemination/Paper_replication_package/Data/test_performance/obs_test_performance_greedy.csv")

# Correction to calculcate approximate equivalent number of training observations
n_train = 15757685 
correction_flat = n_train / NROW(flat)
correction_greedy = n_train / NROW(greedy) 
if(correction_greedy == correction_flat){
    correction = correction_flat
} else {
    stop("Corrections for flat and greedy do not match!")
}

# ==== Data cleaning bits ====
# Stats to use
all_stats = greedy %>% 
  pivot_longer(c(acc, precision, recall, f1), names_to = "stat") %>%
  group_by(stat) %>%
  summarise(
    value = mean(value, na.rm = TRUE),
    n = n()
  ) %>%
  mutate(
    stat = case_when(
      stat == "acc" ~ "Accuracy",
      stat == "f1" ~ "F1 score",
      stat == "precision" ~ "Precision",
      stat == "recall" ~ "Recall"
    )
  )

to_long = function(x){
    x  %>% 
        pivot_longer(c(acc, precision, recall, f1), names_to = "stat") %>%
        group_by(hisco_1, stat) %>%
            summarise(
                value = mean(value, na.rm = TRUE),
                n = n()
            ) %>% 
        ungroup() %>%
        mutate(
            pct_of_train = n/sum(n)
        ) %>% 
        group_by(stat) %>% 
        arrange(-pct_of_train) %>% 
        ungroup() %>%
        mutate(pct_cum_sum = cumsum(pct_of_train)) %>%
        mutate(
            stat = case_when(
            stat == "acc" ~ "Accuracy",
            stat == "f1" ~ "F1 score",
            stat == "precision" ~ "Precision",
            stat == "recall" ~ "Recall"
            )
        ) %>%
        mutate(
            n_corrected = n * correction, # Correct for the number of training observations
        )
}

long_greedy = to_long(greedy)
long_flat = to_long(flat)

# ==== frequency_vs_performance ====
frequency_versus_performance = function(long_data, name) {
    q99 = long_data %>% 
        group_by(stat) %>% 
        filter(pct_cum_sum > 0.99) %>% 
        filter(pct_cum_sum == min(pct_cum_sum))

    labels99 = long_data %>% 
        group_by(stat) %>% 
        summarise(
            n_label = quantile(n_corrected, p = 0.95)
        ) %>% 
        mutate(
            y = 0.4,
            label = "99% of\nobservations"
        )

    labels01 = long_data %>% 
        group_by(stat) %>% 
        summarise(
            n_label = quantile(n_corrected, p = 0.20)
        ) %>% 
        mutate(
            y = 0.4,
            label = "1% of\nobservations"
        )
    
    # Disable scientific notation
    options(scipen = 999)

    p1 = long_data %>% 
        drop_na(stat) %>%
        ggplot(aes(n_corrected, value, label = hisco_1)) +
        geom_smooth(se = FALSE, col = colours$red, method = "gam") + 
        geom_point(size = 0.1, shape = 4) + 
        facet_wrap(~stat) + 
        theme_bw() +
        scale_y_continuous(
            labels = scales::percent,
            breaks = seq(0, 1, by = 0.1),
        ) + 
        scale_x_log10() +
        labs(
            x = "N in training",
            y = "Statistic"
        ) + 
        geom_hline(yintercept = 1) + 
        geom_hline(aes(yintercept = value), data = all_stats, lty = 3) +
        geom_vline(aes(xintercept = n_corrected), data = q99, lty = 3) +
        theme(
            axis.text.x = element_text(angle = 90, vjust = 0.5)
        ) + 
        geom_text(
            aes(x = n_label*1.5, y = y, label = label), data = labels99, size = 2,
        ) + 
        geom_text(
            aes(x = n_label*1.5, y = y, label = label), data = labels01, size = 2,
        )
    
    # fnames 
    fname1 = paste0("Project_dissemination/Paper_replication_package/Tables/", name, "_performance_by_frequency_desc_stats.txt")
    fname2 = paste0("Project_dissemination/Paper_replication_package/Figures/", name, "_performance_by_frequency.png")
    fname2_pdf = paste0("Project_dissemination/Paper_replication_package/Figures/", name, "_performance_by_frequency.pdf")

    # Print sum summary stats
    # N HISCO codes above / below 
    cutoff = q99$n %>% unique() 

    sink(fname1)

    plot_data %>% 
    group_by(stat) %>% 
    filter(n >= cutoff) %>%
    count() %>% print()

    sink()

    # Save the plot
    ggsave(
        fname2, 
        plot = p1, 
        height = dims$height, 
        width = dims$width, 
        dpi = 600
    )

    ggsave(
        fname2_pdf, 
        plot = p1, 
        height = dims$height, 
        width = dims$width, 
        dpi = 600
    )

    return(p1)
}

# ==== Performance by SES ====
performance_by_ses = function(long_data, name) {
    ses_data = long_data %>% 
        mutate(
            ses_value = hisco_to_ses(as.numeric(hisco_1), ses = "hiscam_u1")
        )

    p1 = ses_data %>% 
        ggplot(aes(ses_value, value)) + 
        facet_wrap(~stat) + 
        geom_point(size = 0.1, shape = 4)  + 
        geom_smooth(col = colours$red, method = "gam") +
        theme_bw() +
        labs(y = "Statistic", x = "Socio-Economic Score (HISCAM)")

    fname = paste0("Project_dissemination/Paper_replication_package/Figures/", name, "_performance_by_ses.png")
    ggsave(fname, plot = p1, height = dims$height, width = dims$width, dpi = 600)
    fname_pdf = paste0("Project_dissemination/Paper_replication_package/Figures/", name, "_performance_by_ses.pdf")
    ggsave(fname_pdf, plot = p1, height = dims$height, width = dims$width, dpi = 600)

    # Regressions to test ses_value ~ model performance
    regdata = ses_data %>%
        pivot_wider(names_from = stat, values_from = value, id_cols = c(hisco_1, ses_value, n)) %>%
        mutate(
            n = n*correction # To scale to the amount of training data used
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

    etab1 = etable(
        mod1_acc, mod1_f1, mod1_prec, mod1_recall,
        drop = "Constant",
        tex = TRUE,
        fitstat = "n",
        signif.code=NA
    )

    etab2 = etable(
        mod2_acc, mod2_f1, mod2_prec, mod2_recall,
        drop = "Constant",
        tex = TRUE,
        fitstat = "n",
        signif.code=NA
    )

    # Save regression tables
    fname1 = paste0("Project_dissemination/Paper_replication_package/Tables/", name, "_ses_regression_table.txt")

    sink(fname1)
    print("Panel A: Regression of model performance on SES")
    print(etab1)

    print("Panel B: Regression of model performance on SES and log(N)")
    print(etab2)

    sink()

    return(list(plot = p1, etable1 = etab1, etable2 = etab2))
}


# ==== Main - run all ====
frequency_versus_performance(long_flat, "flat")
frequency_versus_performance(long_greedy, "greedy")

ses_flat = performance_by_ses(long_flat, "flat")
ses_greedy = performance_by_ses(long_greedy, "greedy")


