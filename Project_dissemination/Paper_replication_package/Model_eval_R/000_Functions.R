# Functions
# Created:  2025-03-17
# Authors:  Christian Vedel [christian-vs@sam.sdu.dk],
#
# Purpose:  Contains all the functions used the paper 

# ==== Colors ====
blue = "#273a8f"
green = "#2c5c34"
red = "#b33d3d"
orange = "#DE7500"

dims = list(
  width = 8,
  height = 6
)


# ==== read0() ====
# Read several files without the exact same columns
read0 = function(dir){
  require(foreach)
  require(tidyverse)
  fs = list.files(dir)
  fs = fs[grepl(".csv", fs)]
  foreach(f = fs, .combine = "bind_rows") %do% {
    read_csv(paste0(dir,"/",f), guess_max = 100000) %>% mutate(file = f)
  }
}


# ==== Eval functions ====
# Implements R version of the class defined in 'histocc/eval_metrics.py'
# Function to calculate accuracy for a single observation
acc = function(y_true, y_pred, digits = NULL) {
  if (length(y_true) == 0) return(NaN)
  
  if (!is.null(digits)) {
    y_true = substr(y_true, 1, digits)
    y_pred = substr(y_pred, 1, digits)
  }

  unique(y_true) = unique(y_true[!is.na(y_true)])
  unique(y_pred) = unique(y_pred[!is.na(y_pred)])

  pred_in_true = sum(y_pred %in% y_true)
  true_in_pred = sum(y_true %in% y_pred)
  
  max_preds = max(length(y_true), length(y_pred))
  result = (pred_in_true + true_in_pred) / (2 * max_preds)
  
  return(result)
}

# Function to calculate precision for a single observation
prec = function(y_true, y_pred, digits = NULL) {
  if (length(y_true) == 0) return(NaN)
  
  if (!is.null(digits)) {
    y_true = substr(y_true, 1, digits)
    y_pred = substr(y_pred, 1, digits)
  }

  unique(y_true) = unique(y_true[!is.na(y_true)])
  unique(y_pred) = unique(y_pred[!is.na(y_pred)])
  
  if (length(y_pred) == 0) return(0)
  
  pred_in_true = sum(y_pred %in% y_true)
  return(pred_in_true / length(y_pred))
}

# Function to calculate recall for a single observation
recall = function(y_true, y_pred, digits = NULL) {
  if (length(y_true) == 0) return(NaN)
  
  if (!is.null(digits)) {
    y_true = substr(y_true, 1, digits)
    y_pred = substr(y_pred, 1, digits)
  }

  unique(y_true) = unique(y_true[!is.na(y_true)])
  unique(y_pred) = unique(y_pred[!is.na(y_pred)])
  
  true_in_pred = sum(y_true %in% y_pred)
  return(true_in_pred / length(y_true))
}

# Function to calculate F1 score for a single observation
f1_score = function(y_true, y_pred, digits = NULL) {
  precision = prec(y_true, y_pred, digits)
  recall_value = recall(y_true, y_pred, digits)
  
  if (precision + recall_value == 0) return(0)
  
  return(2 * (precision * recall_value) / (precision + recall_value))
}

# Function to print ETA
print_eta = function(iter, max_iter, start_time, current_time) {
    elapsed_time = current_time - start_time
    time_per_iteration = elapsed_time / iter  # Average time per iteration
    remaining_iterations = max_iter - iter
    eta = time_per_iteration * remaining_iterations
    
    cat(sprintf("Iteration: %d/%d (Elapsed: %.1f sec, ETA: %.1f sec)   \r", 
                iter, max_iter, as.numeric(elapsed_time), as.numeric(eta)))
}
