# Production and calibration curves
# Created:  2026-01-04
# Authors:  Christian Vedel [christian-vs@sam.sdu.dk],
#
# Purpose:  Generate production and calibration curves for test and OOD data

# ==== Libraries ====
library(tidyverse)
library(foreach)
library(knitr)
source("Project_dissemination/Paper_replication_package/Model_eval_R/000_Functions.R")

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
        ) %>%
        mutate(
            # Modify to remove "__" suffix if present
            RowID = gsub("__$", "", RowID)
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
                k = k,
                n = n()
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
        scale_y_continuous(labels = scales::percent_format(accuracy = 1)) +
        labs(
            title = paste0("Top-k production curve (any-of-top-k correct)", title_suffix),
            x = "Share of test set predicted (by confidence)",
            y = "Accuracy among predicted",
            color = "k"
        ) +
        facet_wrap(~metric) +
        theme_bw()

    return(list(plot = p, data = prod_df))
}

# Create LaTeX table for top-k performance at key coverage levels
create_topk_performance_table = function(prod_df, target_proportions = c(0.2, 0.4, 0.6, 0.8, 1.0), k_max = 10) {
    # For each target proportion, find the closest actual value in the data
    closest_props = sapply(target_proportions, function(target) {
        prod_df %>%
            distinct(prop_pred) %>%
            mutate(dist = abs(prop_pred - target)) %>%
            slice_min(dist, n = 1) %>%
            pull(prop_pred)
    })
    
    # Filter to closest proportions and reshape
    table_data = prod_df %>%
        filter(prop_pred %in% closest_props) %>%
        select(metric, k, metric_at_prop, prop_pred) %>%
        pivot_wider(
            names_from = k,
            values_from = metric_at_prop
        ) %>%
        mutate(
            improvement = get(as.character(k_max)) - `1`,
            prop_pred = round(prop_pred, 2)
        ) %>%
        arrange(metric, prop_pred)
    
    return(table_data)
}

# ==== Performance by confidence (greedy predictions) =====
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

# ==== Top-k production curve (test set) =====

# Standard test data top-k
df_test_topk = read_csv("Project_dissemination/Paper_replication_package/Data/Intermediate_data/big_files/predictions_test_top_k/obs_test_topk_10.csv")

# Infer k
K_infered = df_test_topk %>%
    mutate(
        `top-k-pos` = as.numeric(`top-k-pos`)
    ) %>%
    summarise(
        K = max(`top-k-pos`, na.rm = TRUE)+1
    ) %>% pull(K)

res = df_test_topk %>%
    compute_cumulative_performance(K = K_infered) %>%
    create_topk_production_curve(title_suffix = " (Test set)")

ggsave(
    "Project_dissemination/Paper_replication_package/Figures/Production_curve_topk_standard.png",
    plot = res$plot,
    width = dims$width,
    height = dims$height
)

# Summary table
topk_table = create_topk_performance_table(res$data, k_max = K_infered)

# Save to LaTeX
sink("Project_dissemination/Paper_replication_package/Tables/Topk_performance_by_coverage.txt", append = FALSE)
cat("Top-k performance at different coverage levels\n\n")
topk_table %>%
    kable(
        format = "latex",
        booktabs = TRUE,
        digits = 3,
        caption = paste0("Performance metrics across k (top-1 to top-", K_infered, ") at different coverage levels. Improvement shows gain from top-1 to top-", K_infered, "."),
        col.names = c("Metric", "Coverage", paste0("k=", 1:K_infered), paste0("Δ (", K_infered, "-1)"))
    ) %>% print()
sink()

# ==== Top k by HISCO code production curves =====
df_test_topk = read_csv("Project_dissemination/Paper_replication_package/Data/Intermediate_data/big_files/predictions_test_top_k/obs_test_topk_10.csv")


hiscos = df_test_topk %>%
    select(starts_with("hisco_")) %>% 
    select(ends_with("_original")) %>% 
    pivot_longer(
        cols = everything(),
        names_to = "hisco_col",
        values_to = "hisco_code"
    ) %>%
    select(-hisco_col) %>%
    count(hisco_code) %>%
    # Fix missing leading zeros if positve
    mutate(
        hisco_code = ifelse(as.numeric(hisco_code) > 0, sprintf("%05d", as.numeric(hisco_code)), hisco_code)
    ) %>%
    filter(n >= 100)

# Loop
for(h in hiscos$hisco_code) {
    df_hisco = df_test_topk %>%
        rowwise() %>%
        filter(
            any(hisco_1_original == h,
                hisco_2_original == h,
                hisco_3_original == h,
                hisco_4_original == h,
                hisco_5_original == h)
        )

    if(nrow(df_hisco) < 100) {
        next
    }
    print(h)

    res = df_hisco %>%
        compute_cumulative_performance(K = K_infered) %>%
        create_topk_production_curve(title_suffix = paste0(" (HISCO: ", h, ", n=", nrow(df_hisco), ")"))

    ggsave(
        paste0("Project_dissemination/Paper_replication_package/Figures/Production_curve_topk_hisco/hisco_", h, ".png"),
        plot = res$plot,
        width = dims$width,
        height = dims$height
    )
}

# ==== Top k by lang =====
for(l in df_test_topk$lang %>% unique()) {
    df_lang = df_test_topk %>%
        filter(lang == l)

    print(l)

    res = df_lang %>%
        compute_cumulative_performance(K = K_infered) %>%
        create_topk_production_curve(title_suffix = paste0(" (Lang: ", l, ", n=", nrow(df_lang), ")"))

    ggsave(
        paste0("Project_dissemination/Paper_replication_package/Figures/Production_curve_topk_lang/lang_", l, ".png"),
        plot = res$plot,
        width = dims$width,
        height = dims$height
    )
}
