# Descriptive statistics
# Created:  2025-03-20
# Authors:  Christian Vedel [christian-vs@sam.sdu.dk],
#
# Purpose:  Eval stats from ood data

# ==== Libraries ====
library(tidyverse)
library(foreach)
library(knitr)
source("Project_dissemination/Paper_replication_package/Model_eval_R/000_Functions.R")

# ==== Load data ====
files = list.files("Project_dissemination/Paper_replication_package/Data/Intermediate_data/big_files/predictions_ood", full.names = TRUE)

ood_performance = foreach(f = files, .combine = "bind_rows") %do% {
    if(f %in% c(
        "Data/OOD_data/Predictions/Manually_checked"
    )) {
        return(NULL)  # Skip these files
    }

    res = read_csv(f, show_col_types = FALSE) %>%
        mutate(file = f)

    if(all(is.na(res$acc))) {
        return(NULL)  # Skip files with no accuracy data
    }
        
    res %>%
        mutate(file = gsub("Project_dissemination/Paper_replication_package/Data/Intermediate_data/big_files/predictions_ood/", "", file)) %>%
        select(-`...1`, -occ1, -acc, -precision, -recall, -f1) %>%
        pivot_longer(
            cols = acc_1:f1_5,
            names_to = c("statistic", "digits"),
            values_to = "value",
            names_pattern = "(.*)_(\\d+)"
        ) %>%
        group_by(file, statistic, digits) %>%
        summarise(
            value = mean(value, na.rm = TRUE),
            file = file[1],
            n = n()
        )
}

res1 = ood_performance %>% 
    pivot_wider(
        names_from = digits,
        values_from = value
    ) %>%
    mutate(tmp = -`1`) %>%
    arrange(statistic, tmp, file) %>%
    select(-tmp)

# Manually checked files
files = list.files("Project_dissemination/Paper_replication_package/Data/Intermediate_data/big_files/Manually_checked", full.names = TRUE, pattern = "\\.csv$")

ood_performance_manual = foreach(f = files, .combine = "bind_rows") %do% {
    res = read_csv2(f, show_col_types = FALSE) %>%
        mutate(file = f)
        
    res %>%
        mutate(file = gsub("Project_dissemination/Paper_replication_package/Data/Intermediate_data/big_files/Manually_checked/", "", file)) %>%
        select(Check_substantial, Check_strict, file, n) %>%
        pivot_longer(
            cols = Check_substantial:Check_strict,
            names_to = c("statistic"),
            values_to = "value"
        ) %>%
        group_by(file, statistic) %>%
        summarise(
            value = weighted.mean(value, n, na.rm = TRUE),
            file = file[1],
            n = n()
        )
}

res2 = ood_performance_manual %>%
    mutate(statistic = recode(statistic, Check_substantial = "substantial", Check_strict = "strict")) %>%
    arrange(statistic, file) %>%
    pivot_wider(
        names_from = statistic,
        values_from = value
    ) %>%
    mutate(file = gsub("Project_dissemination/Paper_replication_package/Data/Intermediate_data/big_files/Manually_checked/", "", file))

# ==== Manual check of DWI and UK Bankruptcies ====
# Seeded simple random samples of unique strings from the DWI and UK
# Bankruptcies data (N_PER_SOURCE unique strings per source).
# - If the sample does not exist, it is drawn and saved for labelling.
# - If it exists, it is redrawn and checked to be identical.
# - The labelled sample (correct_hisco_1..5 filled in) must be saved as
#   'OOD_manual_check_labelled.csv' in the same folder. Otherwise an error is thrown.
N_PER_SOURCE = 500
SEED = 20 # Reset before each source's draw
manual_sources = tibble(
  source = c("DWI", "UK Bankruptcies"),
  file   = c("predictions_EN_dwi.csv", "predictions_EN_Bankruptcy_KornLacroix.csv")
)
dir_pred   = "Project_dissemination/Paper_replication_package/Data/Intermediate_data/big_files/predictions_ood"
dir_manual = "Project_dissemination/Paper_replication_package/Data/Intermediate_data/Manual_check_samples"
f_sample   = file.path(dir_manual, "OOD_manual_check_sample.csv")
f_labelled = file.path(dir_manual, "OOD_manual_check_labelled.csv")

# Pads HISCO codes to 5 digits (keeps -1, -2, -3 and blanks as is)
pad_hisco = function(x) {
  x = str_trim(as.character(x))
  x = ifelse(x %in% c("", "nan", "NA") | is.na(x), "", x)
  ifelse(str_detect(x, "^\\d+$"), str_pad(x, 5, pad = "0"), x)
}

key = read_csv("Data/Key.csv", col_types = cols(.default = "c")) %>%
  transmute(hisco = pad_hisco(hisco), desc = en_hisco_text) %>%
  filter(hisco != "") %>%
  distinct(hisco, .keep_all = TRUE)

# One row per unique string
build_frame = function(f, source_name) {
  d = read_csv(
    file.path(dir_pred, f),
    col_select = c(occ1, hisco_1, desc_1, hisco_2, desc_2, hisco_1_original),
    col_types = cols(.default = "c")
  ) %>%
    mutate(
      occupation = str_remove(occ1, "^[a-z]{2}\\[SEP\\]"),
      across(c(hisco_1, hisco_2, hisco_1_original), pad_hisco),
      across(c(desc_1, desc_2), ~ ifelse(is.na(.x) | .x == "nan", "", .x))
    ) %>%
    # Missing occupations ("nan" strings) are not labelled
    filter(!is.na(occupation), str_trim(occupation) != "", occupation != "nan")

  # Collapse to unique strings. Predictions are deterministic per string;
  # source codes can differ across rows, so they are listed most frequent first.
  d %>%
    group_by(occupation) %>%
    summarise(
      n = n(),
      occc_hisco_1 = hisco_1[1], occc_desc_1 = desc_1[1],
      occc_hisco_2 = hisco_2[1], occc_desc_2 = desc_2[1],
      source_codes = list(codes_by_freq(hisco_1_original)),
      .groups = "drop"
    ) %>%
    mutate(
      source_hisco = map_chr(source_codes, ~ names(.x)[1]),
      source_hisco_all = map_chr(source_codes, ~ paste0(names(.x), " (", round(.x, 2), ")", collapse = "; "))
    ) %>%
    select(-source_codes) %>%
    left_join(key %>% rename(source_desc = desc), by = c("source_hisco" = "hisco")) %>%
    mutate(source = source_name)
}

# Shares of each code, most frequent first (ties: alphabetical)
codes_by_freq = function(x) {
  tab = table(x)
  tab = tab[order(-tab, names(tab))]
  setNames(as.numeric(tab) / sum(tab), names(tab))
}

sample_frame = function(frame) {
  set.seed(SEED)
  frame %>%
    slice_sample(n = N_PER_SOURCE)
}

frames = map2(manual_sources$file, manual_sources$source, build_frame)

manual_sample = map(frames, sample_frame) %>%
  bind_rows() %>%
  mutate(
    id = 1:n(),
    correct_hisco_1 = "", correct_hisco_2 = "", correct_hisco_3 = "",
    correct_hisco_4 = "", correct_hisco_5 = "",
    comment = ""
  ) %>%
  select(
    id, source, n, occupation,
    occc_hisco_1, occc_desc_1, occc_hisco_2, occc_desc_2,
    source_hisco, source_desc, source_hisco_all,
    starts_with("correct_hisco_"), comment
  )

if (!file.exists(f_sample)) {
  dir.create(dir_manual, showWarnings = FALSE, recursive = TRUE)
  manual_sample %>% write_excel_csv2(f_sample, na = "")

  # Full frames with a sampled flag (documents the population sampled from)
  bind_rows(frames) %>%
    mutate(sampled = paste(source, occupation) %in% paste(manual_sample$source, manual_sample$occupation)) %>%
    select(source, occupation, n, source_hisco, source_hisco_all, occc_hisco_1, occc_hisco_2, sampled) %>%
    write_excel_csv2(file.path(dir_manual, "OOD_manual_check_sampling_frame.csv"), na = "")

  frame_sizes = bind_rows(frames) %>% count(source, name = "n_unique_strings")
  md5 = tools::md5sum(file.path(dir_pred, manual_sources$file))
  sink(file.path(dir_manual, "README.txt"))
  cat("Manual check sample of unique OOD strings (DWI and UK Bankruptcies)\n")
  cat("Created by Model_eval_R/005_OOD_performance.R on", as.character(Sys.Date()), "\n")
  cat(R.version.string, "\n\n")
  cat("Unique strings sampled per source (simple random sample):", N_PER_SOURCE, "\n")
  cat("Seed:", SEED, "(set before each source's draw)\n\n")
  cat("Input files (md5):\n")
  cat(paste0("  ", basename(names(md5)), "  ", md5, collapse = "\n"), "\n\n")
  cat("Sampling frame (unique strings, 'nan' / empty occupations excluded):\n")
  print(as.data.frame(frame_sizes), row.names = FALSE)
  cat("\nFill in correct_hisco_1..5 (use -1 for no occupation) and comment,\n")
  cat("and save as OOD_manual_check_labelled.csv (';'-separated) in this folder.\n")
  sink()
  cat("Manual check sample written to", f_sample, "\n")
} else {
  existing = read_csv2(f_sample, col_types = cols(.default = "c"))
  if (!identical(existing$occupation, manual_sample$occupation) || !identical(existing$source, manual_sample$source)) {
    stop(paste0(f_sample, " does not match the seeded sample. Has the sampling or the prediction files changed?"))
  }
}

if (!file.exists(f_labelled)) {
  stop(paste0(
    "Manually labelled file not found: ", f_labelled, "\n",
    "Label ", f_sample, " (fill in correct_hisco_1..5) and save it under that name."
  ))
}

# ==== Score manual labels ====
labelled = read_csv2(f_labelled, col_types = cols(.default = "c")) %>%
  mutate(
    id = as.integer(id),
    across(starts_with("correct_hisco_"), pad_hisco)
  )

if (!identical(as.integer(labelled$id), manual_sample$id) ||
    !identical(str_trim(labelled$occupation), str_trim(manual_sample$occupation))) {
  stop(paste0(f_labelled, " does not have the same rows (id, occupation) as ", f_sample))
}
if (any(labelled$correct_hisco_1 == "")) {
  stop(paste0("correct_hisco_1 is empty for ids: ", paste(labelled$id[labelled$correct_hisco_1 == ""], collapse = ", ")))
}
codes_used = labelled %>% select(starts_with("correct_hisco_")) %>% unlist() %>% unique()
codes_bad = setdiff(codes_used[codes_used != ""], key$hisco)
if (length(codes_bad) > 0) {
  stop(paste0("Labelled codes not in the HISCO key: ", paste(codes_bad, collapse = ", ")))
}

# Source labels can differ across rows with the same string. The source is
# scored by the share-weighted accuracy of its codes (shares from source_hisco_all).
score_source = function(correct, source_hisco_all, digits) {
  parts = str_split(source_hisco_all, "; ")[[1]]
  codes = str_remove(parts, " \\(.*\\)$")
  shares = as.numeric(str_extract(parts, "(?<=\\()[0-9.]+(?=\\))"))
  shares = shares / sum(shares)
  sum(shares * map_dbl(codes, ~ acc(correct, .x, digits = digits)))
}

manual_scores = labelled %>%
  select(id, source, occupation, starts_with("correct_hisco_")) %>%
  left_join(manual_sample %>% select(id, n, occc_hisco_1, occc_hisco_2, source_hisco_all), by = "id") %>%
  mutate(
    n = as.numeric(n),
    correct = pmap(list(correct_hisco_1, correct_hisco_2, correct_hisco_3, correct_hisco_4, correct_hisco_5), ~ setdiff(c(...), "")),
    model = map2(occc_hisco_1, occc_hisco_2, ~ setdiff(c(.x, .y), "")),
    acc_model = map2_dbl(correct, model, ~ acc(.x, .y, digits = 5)),
    acc_source = map2_dbl(correct, source_hisco_all, ~ score_source(.x, .y, digits = 5))
  )

res3 = manual_scores %>%
  group_by(source) %>%
  summarise(
    strings = n(),
    model = mean(acc_model),
    model_obs = weighted.mean(acc_model, n),
    source_labels = mean(acc_source),
    source_labels_obs = weighted.mean(acc_source, n),
    .groups = "drop"
  )


# Latex in three panels printed to Tables/ood_performance.tex
# Panel A: Manually checked
sink("Project_dissemination/Paper_replication_package/Tables/ood_performance.txt")
res2 %>%
    mutate_if(is.numeric, ~ round(., 3)) %>%
    select(file, n, substantial, strict) %>%
    kable(
        format = "latex",
        booktabs = TRUE,
        caption = "Manually checked OOD data performance"
    ) %>% print()

# Panel B: Data which is already HISCO coded
res1 %>%
    mutate_if(is.numeric, ~ round(., 3)) %>%
    filter(statistic == "acc") %>%
    kable(
        format = "latex",
        booktabs = TRUE,
        caption = "OOD data performance",
        col.names = c("File", "Statistic", "n", "1 digit", "2 digits", "3 digits", "4 digits", "5 digits")
    ) %>% print()

# Panel C: Manual check of DWI and UK Bankruptcies (agreement with manual labels)
res3 %>%
    mutate_if(is.numeric, ~ round(., 3)) %>%
    kable(
        format = "latex",
        booktabs = TRUE,
        caption = "Manually checked DWI and UK Bankruptcies: agreement with manual labels",
        col.names = c("Dataset", "Strings", "OccCANINE", "OccCANINE (obs. weighted)", "Source", "Source (obs. weighted)")
    ) %>% print()

sink()



