# Evaluate model
# Created:    2023-05-23
# Authors:    Christian Vedel [christian-vs@sam.sdu.dk],
#
# Purpose:    Reads predicted labels and produces evaluation results
#
# Output:     Various plots and figures 

# ==== Libraries ====
library(tidyverse)
library(foreach)

# ==== Read data ====
predictions = read_csv("Data/Predictions/XML_RoBERTa_Multilingual_sample_size_6_lr_2e-05_batch_size_32.csv")

# ==== CorrectHISCOChar ====
# Corrects HISCO with leading zeros
CorrectHISCOChar = function(x){
  res = ifelse(nchar(x) == 4, paste0(0,x), x)
  return(res)
}

# ==== EvalIt ====
NoNA_sort = function(x){
  x = x[!is.na(x)]
  x = sort(x)
  if(length(x)==0){x=-1}
  return(x)
}

NAtoZero = function(x){
  ifelse(is.na(x), 0, x)
}

acc = function(x, y){
  if(length(x) != length(y)){
    return(FALSE)
  }
  return(all(x == y))
}


EvalIt = function(data){
  capN = NROW(predictions)
  
  # Run stats
  data %>%
    rowwise() %>%
    mutate(
      tmp = list(NoNA_sort(c(hisco_1, hisco_2, hisco_3, hisco_4, hisco_5))),
      tmp_pred0 = list(NoNA_sort(c(hisco_pred1_0, hisco_pred2_0, hisco_pred3_0, hisco_pred4_0, hisco_pred5_0))),
      tmp_pred1 = list(NoNA_sort(c(hisco_pred1_1, hisco_pred2_1, hisco_pred3_1, hisco_pred4_1, hisco_pred5_1)))
    ) %>% 
    mutate(
      # Accuracy
      acc0 = acc(unlist(tmp), unlist(tmp_pred0)),
      acc1 = acc(unlist(tmp), unlist(tmp_pred1)),
      acc01 = acc(unlist(tmp_pred0), unlist(tmp_pred1)),
      
      # Precicision (how often is the predicted in the true set)
      prc0 = mean(unlist(tmp_pred0) %in% unlist(tmp)),
      prc1 = mean(unlist(tmp_pred1) %in% unlist(tmp)),
      
      # Precicision (how often is the true set in the predicted set)
      rcl0 = mean(unlist(tmp) %in% unlist(tmp_pred0)),
      rcl1 = mean(unlist(tmp) %in% unlist(tmp_pred1)),
      
    ) %>% 
    mutate(
      f10 = 2*prc0*rcl0/(prc0+rcl0),
      f11 = 2*prc1*rcl1/(prc1+rcl1)
    ) %>% 
    mutate(
      f10 = NAtoZero(f10),
      f11 = NAtoZero(f11)
    ) %>% 
    select(
      acc0,
      acc1,
      prc0,
      prc1,
      rcl0,
      rcl1,
      f10,
      f11
    ) %>% 
    ungroup() %>% 
    mutate(
      pct_obs = n() / capN
    ) %>% 
    summarise_all(mean)
   
}

# ==== Run for different subsets ====
# Overall
overall = predictions %>% 
  EvalIt() %>% 
  mutate(
    stat = "all",
    subset = "Overall"
  )

# Languages
lang = foreach(l = unique(predictions$lang), .combine = "bind_rows") %do% {
  predictions %>% 
    filter(
      lang == l
    ) %>% 
    EvalIt() %>% 
    mutate(
      stat = as.character(l),
      subset = "Languages"
    )
}

# By major category
major = foreach(hisco = 0:9, .combine = "bind_rows") %do% {
  predictions %>% 
    filter(hisco_1 != -1) %>% 
    mutate(
      hisco_tmp = CorrectHISCOChar(as.character(hisco_1))
    ) %>% 
    filter(
      substr(hisco_tmp, 1, 1) %in% hisco
    ) %>% 
    EvalIt() %>% 
    mutate(
      stat = as.character(hisco),
      subset = "Major category"
    )
}

# By ses
predictions = predictions %>%
  mutate(
    occ_rank = hisco::hisco_to_ses(hisco_1), ses = "hisclass"
  )

all_ranks = sort(na.omit(unique(predictions$occ_rank)))

ses = foreach(s = all_ranks, .combine = "bind_rows") %do% {
  predictions %>% 
    filter(hisco_1 != -1) %>% 
    filter(occ_rank == s) %>% 
    EvalIt() %>% 
    mutate(
      stat = as.character(s),
      subset = "By hisclass"
    )
}

# By Year
years = sort(na.omit(unique(predictions$Year)))

year = foreach(y = years, .combine = "bind_rows") %do% {
  predictions %>% 
    filter(Year == y) %>% 
    EvalIt() %>% 
    mutate(
      stat = as.character(y),
      subset = "Year"
    )
}

# By n occupations
predictions = predictions %>% 
  rowwise() %>% 
  mutate(
    noccs = length(NoNA_sort(c(hisco_1, hisco_2, hisco_3, hisco_4, hisco_5)))
  )

noccs = foreach(i = unique(predictions$noccs), .combine = "bind_rows") %do% {
  predictions %>% 
    filter(noccs == i) %>% 
    EvalIt() %>% 
    mutate(
      stat = as.character(i),
      subset = "Number of occupations"
    )
}

# Merge together
eval_combined = overall %>% 
  bind_rows(lang) %>% 
  bind_rows(major) %>% 
  bind_rows(ses) %>% 
  bind_rows(year) %>% 
  bind_rows(noccs)

eval_combined %>% write_csv2("Project_dissemination/Plots/Eval_stats.csv")
