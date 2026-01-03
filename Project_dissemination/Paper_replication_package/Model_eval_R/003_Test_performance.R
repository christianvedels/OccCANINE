# Test model performance
# Created:  2025-04-10
# Authors:  Christian Vedel [christian-vs@sam.sdu.dk],
#
# Purpose:  Make pretty tables of test data performance for the paper

# ==== Libraries ====
library(tidyverse)
library(foreach)
library(knitr)
source("Project_dissemination/Paper_replication_package/Model_eval_R/000_Functions.R")

# ==== Load data ====
files = list.files("Project_dissemination/Paper_replication_package/Data/Intermediate_data/test_performance", full.names = TRUE)
files = files[!grepl("Data/Intermediate_data/test_performance/lang", files)]  # Exclude language folder
files = files[!grepl("Data/Intermediate_data/test_performance/source", files)]  # Exclude source folder
foreach(f = files, .combine = "bind_rows") %do% {
    read_csv(f, show_col_types = FALSE) %>%
        mutate(file = f) %>%
        mutate(file = gsub("Project_dissemination/Paper_replication_package/Data/Intermediate_data/test_performance/", "", file)) %>%
        mutate(
            digits = case_when(
                grepl("digits_1", file) ~ 1,
                grepl("digits_2", file) ~ 2,
                grepl("digits_3", file) ~ 3,
                grepl("digits_4", file) ~ 4,
                grepl("digits_5", file) ~ 5
            )
        )
} -> test_performance


# ==== Data preparation ====
performance_table = test_performance %>% pivot_longer(
    cols = c("accuracy", "precision", "recall", "f1"), 
    names_to = "statistic", 
    values_to = "value"
) %>% 
    select(-file) %>%
    pivot_wider(
        names_from = "digits", 
        values_from = "value"
    ) %>%
    select(
        prediction_type, statistic, lang,
        `1`, `2`, `3`, `4`, `5`,
        n
    ) %>%
    arrange(
        case_when(
            prediction_type == "greedy" ~ 1,
            prediction_type == "flat" ~ 2,
            prediction_type == "full" ~ 3,
        )
    ) %>%
    mutate(
        lang = ifelse(lang == "known", "With lang. info.", "Without lang. info."),
    ) %>%
    mutate(
        # Rename to 'fast' 'good' and 'full'
        prediction_type = case_when(
            prediction_type == "greedy" ~ "good",
            prediction_type == "flat" ~ "fast",
            prediction_type == "full" ~ "full"
        ),
    )
# Make LaTeX table with knitr::kable
sink("Project_dissemination/Paper_replication_package/Tables/Test_performance.txt", append = FALSE)
cat("\nTest set performance by prediction type and number of digits")
cat("\nTest performance with language information\n")
performance_table %>%
    filter(lang == "With lang. info.") %>%
    filter(prediction_type %in% c("good", "fast")) %>%
    select(-lang) %>%
    kable(
        format = "latex",
        booktabs = TRUE,
        digits = 3,
        caption = "Test set performance by prediction type and number of digits"
    ) %>% print()

cat("\n")
cat("Test performance without language information:\n")
performance_table %>%
    filter(lang == "Without lang. info.") %>%
    filter(prediction_type %in% c("good", "fast")) %>%
    select(-lang) %>%
    kable(
        format = "latex",
        booktabs = TRUE,
        digits = 3,
        caption = "Test set performance by prediction type and number of digits (without language information)"
    ) %>% print()
sink()

# Lang info performance boost
sink("Project_dissemination/Paper_replication_package/Tables/Test_performance_lang_info_boost.txt", append = FALSE)
cat("\nTest set performance boost with language information\n")
test_performance %>%
    filter(digits == 5) %>%
    filter(prediction_type %in% c("flat", "greedy")) %>%
    arrange(
        case_when(
            prediction_type == "greedy" ~ 1,
            prediction_type == "flat" ~ 2,
            prediction_type == "full" ~ 3
        )
    ) %>%
    select(prediction_type, lang, accuracy, precision, recall, f1, n) %>%
    pivot_longer(
        cols = c(accuracy, precision, recall, f1),
        names_to = "statistic",
        values_to = "value"
    ) %>%
    pivot_wider(
        names_from = lang,
        values_from = value
    ) %>%
    mutate(dif = known - unk) %>%
    pivot_longer(
        c(known, unk, dif),
        names_to = c("lang"),
    ) %>%
    pivot_wider(
        names_from = statistic,
        values_from = value
    ) %>%
    select(
        prediction_type,
        lang, 
        accuracy, precision, recall, f1, n
    ) %>%
    kable(
        format = "latex",
        booktabs = TRUE,
        digits = 3
    ) %>% print()
sink()

# Save optimal performance
test_performance %>%
    filter(digits == 5) %>%
    filter(prediction_type %in% c("greedy")) %>%
    filter(lang == "known") %>%
    write_csv(
        "Project_dissemination/Paper_replication_package/Data/Intermediate_data/test_performance_greedy_wlang.csv"
    )


# ==== Functions ====
# Function to create calibration curve
create_calibration_curve = function(data, lang_filter = NULL, file_filter = NULL, title_suffix = "") {
    # Filter by language if specified
    if (!is.null(lang_filter)) {
        data = data %>% filter(lang == lang_filter)
    }

    # Filter by file if specified
    if (!is.null(file_filter)) {
        data = data %>% filter(file == file_filter)
    }

    if(all(is.na(data$acc))) {
        warning("No accuracy data available for the specified filters.")
        return(NULL)
    }
    
    # Create plot
    p = data %>%
        pivot_longer(
            cols = c("acc", "precision", "recall", "f1"),
            names_to = "metric",
            values_to = "value"
        ) %>%
        mutate(conf = as.numeric(conf), value = as.numeric(value)) %>%
        filter(conf <= 1) %>%  # Filter out invalid confidence values
        ggplot(aes(x = conf, y = value)) +
        geom_smooth() + 
        geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "red") +
        scale_x_continuous(labels = scales::percent_format(accuracy = 1), limits = c(0, 1)) +
        scale_y_continuous(labels = scales::percent_format(accuracy = 1), limits = c(0, 1)) +
        labs(
            title = paste0("Calibration curve of OccCANINE predictions", title_suffix),
            x = "Predicted confidence",
            y = "Observed performance"
        ) +
        facet_wrap(~metric) +
        theme_bw()
    
    return(p)
}

# Function to create production curve
create_production_curve = function(data, lang_filter = NULL, file_filter = NULL, title_suffix = "") {
    # Filter by language if specified
    if (!is.null(lang_filter)) {
        data = data %>% filter(lang == lang_filter)
    }

    # Filter by file if specified
    if (!is.null(file_filter)) {
        data = data %>% filter(file == file_filter)
    }

    if(all(is.na(data$acc))) {
        warning("No accuracy data available for the specified filters.")
        return(NULL)
    }
    
    # Prepare production curve data
    prod_df = data %>%
        pivot_longer(
            cols = c("acc", "precision", "recall", "f1"),
            names_to = "metric",
            values_to = "acc"
        ) %>%
        group_by(metric) %>%
        arrange(desc(conf)) %>%
        mutate(
            idx = row_number(),
            cum_correct = cumsum(acc),
            metric_at_prop = cum_correct / idx
        ) %>% 
        mutate(
            n = n(),
            prop_pred = idx / n
        ) %>%
        ungroup()
    
    # Calculate minimum y-axis value based on mean performance
    min_y = data %>%
        pivot_longer(
            cols = c("acc", "precision", "recall", "f1"),
            names_to = "metric",
            values_to = "acc"
        ) %>% 
        group_by(metric) %>%
        summarise(value = mean(acc)) %>%
        pull(value) %>% 
        min()
    
    # Create plot
    p = ggplot(prod_df, aes(x = prop_pred, y = metric_at_prop)) +
        geom_line(color = "steelblue") +
        facet_wrap(~metric) +
        scale_x_continuous(labels = scales::percent_format(accuracy = 1), limits = c(0, 1)) +
        scale_y_continuous(labels = scales::percent_format(accuracy = 1), limits = c(min_y, 1)) +
        labs(
            title = paste0("Production curve by confidence threshold", title_suffix),
            x = "Share of test set predicted (by confidence)",
            y = "Accuracy among predicted"
        ) +
        theme_bw()
    
    return(p)
}

# ==== Performance by confidence =====
individual_obs = read_csv("Project_dissemination/Paper_replication_package/Data/Intermediate_data/big_files/obs_test_performance_greedy.csv", show_col_types = FALSE)

# Overall calibration curve
p1 = create_calibration_curve(individual_obs)
ggsave(
    "Project_dissemination/Paper_replication_package/Figures/Calibration_curve.png",
    plot = p1,
    width = dims$width,
    height = dims$height
)

# Overall production curve
p2 = create_production_curve(individual_obs)
ggsave(
    "Project_dissemination/Paper_replication_package/Figures/Production_curve.png",
    plot = p2,
    width = dims$width,
    height = dims$height
)

# Production curves by language
for(l in individual_obs$lang %>% unique()) {
    p_lang = create_production_curve(
        individual_obs, 
        lang_filter = l, 
        title_suffix = paste0(" (", l, ")")
    )
    
    ggsave(
        paste0("Project_dissemination/Paper_replication_package/Figures/Production_curve_by_lang/", l, ".png"),
        plot = p_lang,
        width = dims$width,
        height = dims$height
    )
}

# Calibration curves by language
for(l in individual_obs$lang %>% unique()) {
    p_lang = create_calibration_curve(
        individual_obs, 
        lang_filter = l, 
        title_suffix = paste0(" (", l, ")")
    )
    
    ggsave(
        paste0("Project_dissemination/Paper_replication_package/Figures/Calibration_curve_by_lang/", l, ".png"),
        plot = p_lang,
        width = dims$width,
        height = dims$height
    )
}

# ==== OOD production/calibration curves =====
dir = "Project_dissemination/Paper_replication_package/Data/Intermediate_data/big_files/predictions_ood"
ood_files = list.files(dir)
ood_obs = foreach(f = ood_files, .combine = "bind_rows") %do% {
    print(f)
    read_csv(paste0(dir, "/", f), show_col_types = FALSE, col_types = "ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc") %>%
        mutate(file = f)
}

# Production curves for OOD files
for(f in ood_obs$file %>% unique()) {
    p_ood = create_production_curve(
        ood_obs, 
        file_filter = f, 
        title_suffix = paste0(" (OOD: ", gsub("Project_dissemination/Paper_replication_package/Data/Intermediate_data/big_files/predictions_ood/", "", f), ")")
    )
    
    if(is.null(p_ood)) {
        next
    }

    ggsave(
        paste0("Project_dissemination/Paper_replication_package/Figures/Production_curve_ood/", gsub("Project_dissemination/Paper_replication_package/Data/Intermediate_data/big_files/predictions_ood/", "", f), ".png"),
        plot = p_ood,
        width = dims$width,
        height = dims$height
    )
}

# Calibration curves for OOD files
for(f in ood_obs$file %>% unique()) {
    p_ood = create_calibration_curve(
        ood_obs, 
        file_filter = f, 
        title_suffix = paste0(" (OOD: ", gsub("Project_dissemination/Paper_replication_package/Data/Intermediate_data/big_files/predictions_ood/", "", f), ")")
    )

    if(is.null(p_ood)) {
        next
    }
    
    ggsave(
        paste0("Project_dissemination/Paper_replication_package/Figures/Calibration_curve_ood/", gsub("Project_dissemination/Paper_replication_package/Data/Intermediate_data/big_files/predictions_ood/", "", f), ".png"),
        plot = p_ood,
        width = dims$width,
        height = dims$height
    )
}


# ==== Top-k production curves (cumulative: any-of-top-k correct) ====

compute_cumulative_performance = function(preds_topk, K = 5, digits = 5) {

    if(!all(c("RowID", "conf", "top-k-pos") %in% names(preds_topk))) {
        stop("Expected columns RowID, conf, and top-k-pos in top-k predictions")
    }

    # Ensure types
    preds_topk = preds_topk %>%
        mutate(
            `top-k-pos` = as.numeric(`top-k-pos`),
            conf = as.numeric(conf),
            acc = as.numeric(acc),
            precision = as.numeric(precision),
            recall = as.numeric(recall),
            f1 = as.numeric(f1)
        )

    res = foreach(k = 1:K, .combine = "bind_rows") %do% {
        preds_topk %>%
            filter(`top-k-pos` < k) %>%
            group_by(RowID) %>%
            summarise(
                acc = max(acc, na.rm = TRUE),
                precision = max(precision, na.rm = TRUE),
                recall = max(recall, na.rm = TRUE),
                f1 = max(f1, na.rm = TRUE),
                conf = sum(conf, na.rm = TRUE),
                k = k
            )
    } %>% ungroup()

    return(res)
}

create_topk_production_curve = function(topk_perf, title_suffix = "") {
    prod_df = topk_perf %>%
        pivot_longer(
            cols = c("acc", "precision", "recall", "f1"),
            names_to = "metric",
            values_to = "acc"
        ) %>%
        mutate(k = factor(k, levels = sort(unique(k)))) %>%
        group_by(metric, k) %>%
        arrange(desc(conf), .by_group = TRUE) %>%
        mutate(
            idx = row_number(),
            cum_correct = cumsum(acc),
            metric_at_prop = cum_correct / idx
        ) %>% 
        mutate(
            n = n(),
            prop_pred = idx / n
        ) %>%
        ungroup()

    # Calculate minimum y-axis value based on mean performance
    min_y = prod_df %>% 
        group_by(metric) %>%
        summarise(value = mean(acc)) %>%
        pull(value) %>% 
        min()

    p = ggplot(prod_df, aes(x = prop_pred, y = metric_at_prop, color = k)) +
        geom_line() +
        scale_color_manual(values = scales::alpha(colours$red, seq(1, 0.2, length.out = nlevels(prod_df$k)))) +
        scale_x_continuous(labels = scales::percent_format(accuracy = 1), limits = c(0, 1)) +
        scale_y_continuous(labels = scales::percent_format(accuracy = 1), limits = c(min_y, 1)) +
        labs(
            title = paste0("Top-k production curve (any-of-top-k correct)", title_suffix),
            x = "Share of test set predicted (by confidence)",
            y = "Accuracy among predicted",
            color = "k"
        ) +
        facet_wrap(~metric) +
        theme_bw()

    return(p)
}

# ==== Top-k production curve (test set) =====
topk_dir = "Project_dissemination/Paper_replication_package/Data/Intermediate_data/big_files/predictions_test_top_k"
topk_files = list.files(topk_dir, full.names = TRUE, pattern = "\\.csv$")


# Standard test data top-k
df_test_topk = read_csv("Project_dissemination/Paper_replication_package/Data/Intermediate_data/big_files/predictions_test_top_k/obs_test_topk_5.csv")

p = df_test_topk %>%
    compute_cumulative_performance(K = 5) %>%
    create_topk_production_curve(title_suffix = " (Test set)")

ggsave(
    "Project_dissemination/Paper_replication_package/Figures/Production_curve_topk_standard.png",
    plot = p,
    width = dims$width,
    height = dims$height
)

# Unique strings test data top-k
df_test_topk = read_csv("Project_dissemination/Paper_replication_package/Data/Intermediate_data/big_files/predictions_test_top_k/obs_test_unique_topk_5.csv")

p = df_test_topk %>%
    compute_cumulative_performance(K = 5) %>%
    create_topk_production_curve(title_suffix = " (Test set - unique strings)")

ggsave(
    "Project_dissemination/Paper_replication_package/Figures/Production_curve_topk_unique.png",
    plot = p,
    width = dims$width,
    height = dims$height
)

# OOD top-k
