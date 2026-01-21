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
        scale_x_continuous(labels = scales::percent_format(accuracy = 0.01), limits = c(0, 1)) +
        scale_y_continuous(labels = scales::percent_format(accuracy = 0.01), limits = c(0, 1)) +
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
        scale_x_continuous(labels = scales::percent_format(accuracy = 0.01), limits = c(0, 1)) +
        scale_y_continuous(labels = scales::percent_format(accuracy = 0.01), limits = c(min_y, 1)) +
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

    # We base cumulative performance sorting by top-1 confidence
    top_1_conf = preds_topk %>%
        filter(`top-k-pos` == 0) %>%
        select(RowID, conf)

    res = foreach(k = 1:K, .combine = "bind_rows") %do% {
        preds_topk %>%
            filter(`top-k-pos` < k) %>%
            group_by(RowID) %>%
            summarise(
                acc = max(acc, na.rm = TRUE),
                precision = max(precision, na.rm = TRUE),
                recall = max(recall, na.rm = TRUE),
                f1 = max(f1, na.rm = TRUE),
                # conf = sum(conf, na.rm = TRUE), # This would not guarantee curve_k is above curve_k-1 since summing can change order
                k = k,
                n = n()
            ) %>%
            left_join(
                top_1_conf, # Use top-1 confidence for ordering to guarantee no reshuffling of confidence ranking
                by = "RowID" 
            )
    } %>% ungroup()

    return(res)
}

create_topk_production_curve = function(prod_df, reporting_freq = 1) {

    prod_df = prod_df %>%
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

    # Reference and filtering
    prod0 = prod_df
    ref = prod_df %>%
        filter(k == 1)
    prod_df = prod_df %>%
        filter(as_numeric_factor(k) %% reporting_freq == 0) %>%
        filter(k != 1)

    # Calculate minimum y-axis value based on mean performance
    min_y = prod_df %>% 
        group_by(metric) %>%
        summarise(value = mean(acc)) %>%
        pull(value) %>% 
        min()

    k_vals = unique(prod_df$k) %>% unique() %>% as_numeric_factor()
    
    # Combine ref and prod_df for unified color scale
    # Put 1 last in levels so k=1 is drawn on top and always visible
    all_levels = c(setdiff(levels(prod_df$k), "1"), 1)
    
    combined_df = bind_rows(
        prod_df %>% mutate(k = factor(k, levels = all_levels)),
        ref %>% mutate(k = factor(1, levels = all_levels))
    )
    
    # Create color vector: red gradient for k>1, black for k=1 (last)
    all_k_vals = c(k_vals, 1)
    color_vals = c(scales::alpha(colours$red, seq(1, by = -1/length(k_vals), length.out = length(k_vals))), "black")
    names(color_vals) = all_k_vals

    p = combined_df %>% 
        ggplot(aes(x = prop_pred, y = metric_at_prop, color = k, group = k)) +
        geom_line() +
        scale_color_manual(values = color_vals) +
        scale_x_continuous(labels = scales::percent_format(accuracy = 0.01), limits = c(0, 1)) +
        scale_y_continuous(labels = scales::percent_format(accuracy = 0.01)) +
        labs(
            x = "Share of test set predicted (by confidence)",
            y = "Accuracy among predicted",
            color = "k"
        ) +
        facet_wrap(~metric) +
        theme_bw()


    return(list(plot = p, data = prod0))
}


average_improvement = function(prod_df, k_max, reporting_freq = 1) {
    # Compute improvements overall
    prod_df_sum = prod_df %>%
        group_by(metric, k) %>%
        summarise(acc = mean(acc), .groups = "drop") %>%
        mutate(
            improvement = acc - min(acc)
        )

    ref = prod_df_sum %>%
        filter(k == 1) %>%
        select(metric, acc) %>%
        rename(ref_acc = acc) %>%
        right_join(prod_df_sum, by = "metric")

    p1 = prod_df_sum %>% ggplot(aes(x = k, y = acc, group = metric)) +
        geom_line(col = colours$red) +
        geom_point(col = colours$red) +
        labs(
            x = "k",
            y = "Performance"
        ) +
        theme_bw() + 
        facet_wrap(~metric) +
        geom_hline(data = ref, aes(yintercept = ref_acc), linetype = "dashed") + 
        scale_y_continuous(labels = scales::percent_format(accuracy = 0.01))

    # Create improved table
    # Get k=1 baseline performance
    baseline = prod_df_sum %>%
        filter(k == 1) %>%
        select(metric, baseline = acc)
    
    # Filter k values by reporting frequency (exclude k=1 from improvements)
    improvement_df = prod_df_sum %>%
        filter(k != 1, as_numeric_factor(k) %% reporting_freq == 0) %>%
        select(metric, k, improvement) %>%
        pivot_wider(
            names_from = k,
            values_from = improvement,
            names_prefix = "k"
        )
    
    # Combine baseline with improvements
    table_data = baseline %>%
        left_join(improvement_df, by = "metric") %>%
        arrange(metric)

    return(list(overall = p1, prod_df_sum = prod_df_sum, table = table_data))
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

prod_df = df_test_topk %>%
    compute_cumulative_performance(K = K_infered)
    
res = prod_df %>%
    create_topk_production_curve()

ggsave(
    "Project_dissemination/Paper_replication_package/Figures/Production_curve_topk_standard.png",
    plot = res$plot,
    width = dims$width,
    height = dims$height
)

# Average improvement plot
res_avg = res$data %>%
    average_improvement(k_max = K_infered, reporting_freq = 1)

ggsave(
    "Project_dissemination/Paper_replication_package/Figures/Average_improvement_topk.png",
    plot = res_avg$overall,
    width = dims$width,
    height = dims$height
)

# Average improvement table
avg_table = res_avg$table

# Save to LaTeX
sink("Project_dissemination/Paper_replication_package/Tables/Average_improvement_topk.txt", append = FALSE)
cat("Average improvement by top-k\n\n")
avg_table %>%
    mutate(
        metric = case_when(
            metric == "acc" ~ "Accuracy",
            metric == "precision" ~ "Precision",
            metric == "recall" ~ "Recall",
            metric == "f1" ~ "F1 Score",
            TRUE ~ metric
        )
    ) %>%
    kable(
        format = "latex",
        booktabs = TRUE,
        digits = 3,
        caption = paste0("Average performance improvement across k. Baseline shows k=1 performance, k2-k", K_infered, " show improvement over baseline."),
        col.names = c("Metric", "Baseline (k=1)", paste0("Δ (k=", 2:K_infered, ")"))
    ) %>% print()
sink()

# ==== Top k by HISCO code production curves =====
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

df_test_topk = df_test_topk %>%
    mutate(
        hisco_1_original = ifelse(as.numeric(hisco_1_original) > 0, sprintf("%05d", as.numeric(hisco_1_original)), hisco_1_original),
        hisco_2_original = ifelse(as.numeric(hisco_2_original) > 0, sprintf("%05d", as.numeric(hisco_2_original)), hisco_2_original),
        hisco_3_original = ifelse(as.numeric(hisco_3_original) > 0, sprintf("%05d", as.numeric(hisco_3_original)), hisco_3_original),
        hisco_4_original = ifelse(as.numeric(hisco_4_original) > 0, sprintf("%05d", as.numeric(hisco_4_original)), hisco_4_original),
        hisco_5_original = ifelse(as.numeric(hisco_5_original) > 0, sprintf("%05d", as.numeric(hisco_5_original)), hisco_5_original)
    )

# Loop
avg_table_by_hisco = foreach(h = hiscos$hisco_code, .combine = "bind_rows") %do% {
    df_hisco = df_test_topk %>%
        rowwise() %>%
        filter(
            any(hisco_1_original == h,
                hisco_2_original == h,
                hisco_3_original == h,
                hisco_4_original == h,
                hisco_5_original == h)
        )

    print(h)

    res = df_hisco %>%
        compute_cumulative_performance(K = K_infered) %>%
        create_topk_production_curve(reporting_freq = 2)

    res$plot

    unique_obs = df_hisco$RowID %>% unique()

    ggsave(
        paste0("Project_dissemination/Paper_replication_package/Figures/Production_curve_topk_hisco/hisco_", h, ".png"),
        plot = res$plot + labs(
            title = paste0("HISCO code ", h, " (n=", length(unique_obs), ")")
        ),
        width = dims$width,
        height = dims$height
    )

    # Average improvement plot
    res_avg = res$data %>%
        average_improvement(k_max = K_infered, reporting_freq = 1)

    ggsave(
        paste0("Project_dissemination/Paper_replication_package/Figures/Average_improvement_topk_hisco/hisco_", h, ".png"),
        plot = res_avg$overall,
        width = dims$width,
        height = dims$height
    )

    # Average improvement table
    avg_table = res_avg$table
    avg_table$n = nrow(df_hisco)
    avg_table$hisco = h

    return(avg_table)
}

# Create summary table for HISCO codes
avg_table_by_hisco %>%
    select(hisco, n, metric, baseline, k2, k5, k10) %>%
    mutate(
        metric = case_when(
            metric == "acc" ~ "Accuracy",
            metric == "precision" ~ "Precision",
            metric == "recall" ~ "Recall",
            metric == "f1" ~ "F1 Score",
            TRUE ~ metric
        )
    ) %>%
    arrange(hisco, metric) %>%
    kable(
        format = "latex",
        booktabs = TRUE,
        digits = 3,
        caption = "Top-k performance improvement by HISCO code. Baseline shows k=1 performance, k2/k5/k10 show improvement over baseline.",
        col.names = c("HISCO", "N", "Metric", "Baseline", "Δ(k=2)", "Δ(k=5)", "Δ(k=10)")
    ) %>%
    write_file("Project_dissemination/Paper_replication_package/Tables/Average_improvement_by_hisco.txt")

# ==== Top k by lang =====
avg_table_by_lang = foreach(l = df_test_topk$lang %>% unique(), .combine = "bind_rows") %do% {
    df_lang = df_test_topk %>%
        filter(lang == l)

    print(l)

    res = df_lang %>%
        compute_cumulative_performance(K = K_infered) %>%
        create_topk_production_curve(reporting_freq = 2)

    unique_obs = df_lang$RowID %>% unique()

    ggsave(
        paste0("Project_dissemination/Paper_replication_package/Figures/Production_curve_topk_lang/lang_", l, ".png"),
        plot = res$plot + labs(
            title = paste0("Language: ", l, " (n=", length(unique_obs), ")")
        ),
        width = dims$width,
        height = dims$height
    )

    # Average improvement plot
    res_avg = res$data %>%
        average_improvement(k_max = K_infered, reporting_freq = 1)

    ggsave(
        paste0("Project_dissemination/Paper_replication_package/Figures/Average_improvement_topk_lang/lang_", l, ".png"),
        plot = res_avg$overall,
        width = dims$width,
        height = dims$height
    )

    # Average improvement table
    avg_table = res_avg$table
    avg_table$n = nrow(df_lang)
    avg_table$lang = l

    return(avg_table)
}

# Create summary table for languages
avg_table_by_lang %>%
    select(lang, n, metric, baseline, k2, k5, k10) %>%
    mutate(
        metric = case_when(
            metric == "acc" ~ "Accuracy",
            metric == "precision" ~ "Precision",
            metric == "recall" ~ "Recall",
            metric == "f1" ~ "F1 Score",
            TRUE ~ metric
        )
    ) %>%
    arrange(lang, metric) %>%
    kable(
        format = "latex",
        booktabs = TRUE,
        digits = 3,
        caption = "Top-k performance improvement by language. Baseline shows k=1 performance, k2/k5/k10 show improvement over baseline.",
        col.names = c("Language", "N", "Metric", "Baseline", "Δ(k=2)", "Δ(k=5)", "Δ(k=10)")
    ) %>%
    write_file("Project_dissemination/Paper_replication_package/Tables/Average_improvement_by_lang.txt")

# ==== Top-k production curves (OOD data) =====
dir_ood_topk = "Project_dissemination/Paper_replication_package/Data/Intermediate_data/big_files/predictions_ood_top_k"
ood_topk_files = list.files(dir_ood_topk)

# Process each OOD file
avg_table_by_ood = foreach(f = ood_topk_files, .combine = "bind_rows") %do% {
    print(f)

    # Read OOD top-k data
    df_ood_topk = read_csv(
        paste0(dir_ood_topk, "/", f),
        show_col_types = FALSE
    )

    # Infer K
    K_infered = df_ood_topk %>%
        mutate(`top-k-pos` = as.numeric(`top-k-pos`)) %>%
        summarise(K = max(`top-k-pos`, na.rm = TRUE) + 1) %>%
        pull(K)

    # Compute cumulative performance
    prod_df = df_ood_topk %>%
        compute_cumulative_performance(K = K_infered)

    # Create production curve
    res = prod_df %>%
        create_topk_production_curve(reporting_freq = 2)

    # Count unique observations
    unique_obs = df_ood_topk$RowID %>% unique()

    # Save production curve plot
    ggsave(
        paste0("Project_dissemination/Paper_replication_package/Figures/Production_curve_topk_ood/", f, ".png"),
        plot = res$plot + labs(
            title = paste0("OOD: ", gsub("predictions_", "", gsub(".csv", "", f)), " (n=", length(unique_obs), ")")
        ),
        width = dims$width,
        height = dims$height
    )

    # Average improvement analysis
    res_avg = res$data %>%
        average_improvement(k_max = K_infered, reporting_freq = 1)

    # Save average improvement plot
    ggsave(
        paste0("Project_dissemination/Paper_replication_package/Figures/Average_improvement_topk_ood/", f, ".png"),
        plot = res_avg$overall + labs(
            title = paste0("OOD: ", gsub("predictions_", "", gsub(".csv", "", f)))
        ),
        width = dims$width,
        height = dims$height
    )

    # Build table with dataset identifier
    avg_table = res_avg$table
    avg_table$n = length(unique_obs)
    avg_table$dataset = gsub("predictions_", "", gsub(".csv", "", f))

    return(avg_table)
}

# Create summary table for OOD datasets
avg_table_by_ood %>%
    select(dataset, n, metric, baseline, k2, k5, k10) %>%
    mutate(
        metric = case_when(
            metric == "acc" ~ "Accuracy",
            metric == "precision" ~ "Precision",
            metric == "recall" ~ "Recall",
            metric == "f1" ~ "F1 Score",
            TRUE ~ metric
        )
    ) %>%
    arrange(dataset, metric) %>%
    kable(
        format = "latex",
        booktabs = TRUE,
        digits = 3,
        caption = "Top-k performance improvement for OOD datasets. Baseline shows k=1 performance, k2/k5/k10 show improvement over baseline.",
        col.names = c("Dataset", "N", "Metric", "Baseline", "Δ(k=2)", "Δ(k=5)", "Δ(k=10)")
    ) %>%
    write_file("Project_dissemination/Paper_replication_package/Tables/Average_improvement_topk_ood.txt")



# NOTE (top-k production curves):
# For each observation, top-k performance is defined as "any-of-top-k correct"
# (i.e., metric_k = max(metric over positions 1..k)). This is monotone in k at
# the observation level: increasing k cannot turn a correct case into incorrect.
# However, production curves depend on *ranking* observations by a confidence
# score. If confidence is aggregated across k (e.g., sum of top-k confidences),
# the ranking can reshuffle as k increases, so early prefixes differ across k
# and curves may cross. To isolate the mechanical effect of increasing k, we
# fix the ordering using top-1 confidence for all k (left_join(top_1_conf)).
# This guarantees a common confidence ranking across k and yields pointwise
# non-decreasing production curves in k (up to ties / missingness).
# In practice, users might be interested in other confidence aggregations
# e.g. sum to reflect total confidence mass over top-k predictions.
