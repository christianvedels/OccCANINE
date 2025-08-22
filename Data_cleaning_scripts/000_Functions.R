# Functions
# Created:  2023-05-23
# Authors:  Christian Vedel [christian-vs@sam.sdu.dk],
#
# Purpose:  Contains all the functions used in data cleaning

# ==== Colors ====
colours = list(
  black = "black",
  blue = "#273a8f",
  green = "#2c5c34",
  red = "#b33d3d",
  orange = "#DE7500"
)

dims = list(
  width = 8,
  height = 6
)


# ==== sub_scandi ====
# Replaces all Scandinavian letters with their standard English alphabet 
# counterpart
# x: Character to clean

sub_scandi = function(x){
  require(stringi)
  x = stri_trans_general(x, "Latin-ASCII")
  return(x)
  
  # Legacy code
  scandi_letters = c("Æ",
                     "æ",
                     "Ø",
                     "ø",
                     "Å",
                     "å",
                     "ö",
                     "Ö",
                     "ä",
                     "Ä",
                     "ñ",
                     "Ó",
                     "ó",
                     "ð",
                     "þ",
                     "ü",
                     "á",
                     "í",
                     "û",
                     "ú")
  
  replacement = c("Ae",
                  "ae",
                  "Oe",
                  "oe",
                  "Aa",
                  "aa",
                  "oe",
                  "Oe",
                  "ae",
                  "Ae",
                  "n",
                  "O",
                  "o",
                  "th",
                  "th",
                  "y",
                  "a",
                  "i",
                  "u",
                  "u"
                  
  )
  
  for(i in 1:length(scandi_letters)){
    x = gsub(
      scandi_letters[i],
      replacement[i],
      x
    )
  }
  
  return(x)
  
}

# ==== sub_scandi_utf ====
# Replaces all Scandinavian letters, when they are read wrong. 
# x: Character to clean
sub_scandi_utf = function(x){
  scandi_letters = c(
    "\xd8",
    "\xf8",
    "\xc6",
    "\xe6",
    "\xc5",
    "\xe5",
    "\xc4",
    "\xe4",
    "\xd6",
    "\xf6"
  )
  
  replacement = c(
    "Ø",
    "ø",
    "Æ",
    "æ",
    "Å",
    "å",
    "Æ",
    "æ",
    "Ø",
    "ø"
  )
  
  for(i in 1:length(scandi_letters)){
    x = gsub(
      scandi_letters[i],
      replacement[i],
      x
    )
  }
  
  return(x)
  
}

# ==== gsub_sev ====
# Like gsub, just with several inputs to sub

# a: What to sub
# b: What to sub with
# x: Data in which to perform sub
gsub_sev = function(a, b, x){
  a = unique(a)
  a = na.omit(a)
  for(i in a){
    x = gsub(i, b, x)
  }
  return(x)
}


# ==== LoadRData ====
# Loads Rdata into an object 
# https://stackoverflow.com/questions/5577221/can-i-load-a-saved-r-object-into-a-new-object-name)

loadRData = function(fileName){
  #loads an RData file, and returns it
  load(fileName)
  get(ls()[ls() != "fileName"])
}

# ==== Validate_split ====
# Prints summary statitics which are relevant to check, that the split was correct
Validate_split = function(x){
  res = x %>% 
    group_by(split) %>% 
    count() %>% 
    ungroup() %>% 
    mutate(pct = n/sum(n))
  print(res)
  return(x)
}

# ==== Keep_only_relevant ====
# Selects only the variables which are common across all data
# x:              Data frame
# other_to_keep:  Other vars to keep

Keep_only_relevant = function(x, other_to_keep = NA){
  relevant_vars = c(
    "Year",
    "RowID",
    "occ1",
    "hisco_1",
    "hisco_2",
    "hisco_3",
    "hisco_4",
    "hisco_5",
    "code1",
    "code2",
    "code3",
    "code4", 
    "code5", 
    "split",
    "lang"
  )
  
  if(!all(is.na(other_to_keep))){ # If any, add other vars to list
    relevant_vars = c(relevant_vars, other_to_keep)
  }
  
  res = x[,which(names(x)%in%relevant_vars)]
  return(res)
}

# ==== write_csv0 ====
# write_csv, but with a message
write_csv0 = function(x, fname, only_save_none_empty = FALSE){
  if(only_save_none_empty){
    if(NROW(x) == 0){
      warning("No data. Did not save", fname)
      return(0)
    }
  }
  cat("\nSaving", fname, "with", NROW(x)/1000,"thousands rows")
  write_csv(x, fname)
}

# ==== ensure_path_exists ====
ensure_path_exists = function(file_path) {
  # Extract the directory part from the file path
  dir_path = dirname(file_path)
  
  # Check if the directory exists; if not, create it
  if (!dir.exists(dir_path)) {
    dir.create(dir_path, recursive = TRUE)
  }
}

# ==== Save_train_val_test ====
# Saves train test val data
# x:        Data containing 'split'
# Name      Name of the data. Will be saved as [Name]_[x].csv, where x is test, train, etc.
# language: Language 'da', 'en', 'nl' or 'se', ...
# dir:      Two options "standard" or "other". Defaults to "standard"

Save_train_val_test = function(x, Name, language = NA, dir = "standard"){
  # Throw error if incorrect language
  valid_languages = c('da', 'en', 'nl', 'se', 'no', 'fr', 'ca', 'es', 'pt', 'gr', 'unk', 'ge', 'is', 'unk', 'it', 'In_data')
  if(!language %in% valid_languages){
    stop("Provide correct language")
  }
  
  the_cats = x$split %>% unique()
  
  # Replace NA in occ string
  x = x %>% 
    mutate(
      occ1 = ifelse(is.na(occ1), " ", occ1)
    )
  
  # Update RowID to contain Name
  x = x %>% 
    mutate(
      RowID = paste0(Name, RowID)
    )
  
  # Handle languages
  if(language == 'In_data'){
    # Check if all langs are valid
    the_langs = unique(x$lang)
    test1 = all(the_langs %in% valid_languages)
    if(!test1){
      # Find culprit
      culprit = the_langs[which(!the_langs %in% valid_languages)]
      stop("Unrecognized language found: '",culprit, "' is not a valid language")
    }
  } else {
    # If lang was an input, then set it as variable
    x = x %>% 
      mutate(
        lang = language
      )
  }
  
  # Add na year if it is missing
  if((!"Year" %in% names(x))){
    x = x %>% 
      mutate(Year = NA)
  }
  
  # Filter data
  # browser()
  x_test = x %>% 
    filter(split == "Test")
  x_val1 = x %>% 
    filter(split == "Val1")
  x_val2 = x %>% 
    filter(split == "Val2")
  x_train = x %>% 
    filter(split == "Train")
  
  # Make file names
  if(dir == "standard"){
    fname_test = paste0("Data/Test_data/", Name, "_test.csv")
    fname_val1 = paste0("Data/Validation_data1/", Name, "_val1.csv")
    fname_val2 = paste0("Data/Validation_data2/", Name, "_val2.csv")
    fname_train = paste0("Data/Training_data/", Name, "_train.csv")
  } else if(dir == "other"){ # For other systems of classification
    fname_test = paste0("Data/Test_data_other/", Name, "_test.csv")
    fname_val1 = paste0("Data/Validation_data1_other/", Name, "_val1.csv")
    fname_val2 = paste0("Data/Validation_data2_other/", Name, "_val2.csv")
    fname_train = paste0("Data/Training_data_other/", Name, "_train.csv")
  } else {
    stop("The 'dir' provided is not implemented")
  }

  # Make sure path exists 
  lapply(
    list(fname_test, fname_val1, fname_val2, fname_train), 
    ensure_path_exists
    )
  
  x_test %>% 
    write_csv0(fname_test, only_save_none_empty = TRUE)
  x_val1 %>% 
    write_csv0(fname_val1, only_save_none_empty = TRUE)
  x_val2 %>% 
    write_csv0(fname_val2, only_save_none_empty = TRUE)
  x_train %>% 
    write_csv0(fname_train, only_save_none_empty = TRUE)
}


# ==== upsample() ====
# Upsamples data approximately by x amount
# x: Data to usample
# n: Factor by which to upsamples (to closest 1000)

upsample = function(x, n, verbose = FALSE){

  target_samples = n*NROW(x)
  missing_samples = target_samples - NROW(x)
  
  while(missing_samples>0){
    x = x %>% 
      sample_n(1000) %>% 
      bind_rows(x)
    
    if(verbose){
      cat("\nAdded 1000 samples")
    }
    missing_samples = target_samples - NROW(x)
  }
  
  return(x)
}

# ==== Concat_strings_lang ====
Concat_strings_lang = function(x, and = "och"){
  if(length(na.omit(x))<=1){
    return(x[1])
  }
  
  for(i in seq(length(x))){
    if(is.na(x[i])){
      break
    }
    if(i==1){
      res = x[i]
    } else {
      res = paste0(res, " ", and, " ", x[i])
    }
  }
  
  if(length(res)!=1){
    print(x)
    stop("Too long")
  }
  
  return(res)
}

# ==== NA_str ====
NA_str = function(x){
  ifelse(is.na(x), " ", as.character(x))
}

# ==== translate ====
# Translates with a key
translate = function(x, key){
  x
}

# ==== Combinations ====
Combinations = function(x, and = "and"){
  require(foreach)
  set.seed(20)

  # Initiating empty frame
  results = foreach(i = seq(10), .combine = "bind_rows") %do% x
  index_sample = sample(seq(NROW(results)))
  
  # Generating random combinations
  results = results %>% 
    mutate(tmp_occ = occ1) %>% 
    mutate(
      occ1 = paste(occ1, and, occ1[index_sample]),
      hisco_2 = hisco_1[index_sample]
    ) %>% 
    mutate(
      hisco_2 = ifelse(hisco_2 == "-1", " ", hisco_2)
    ) %>% 
    filter(hisco_1 != "-1") %>% 
    select(-tmp_occ)
  
  return(results)
}

# ==== Data_summary ====
# Gives a summary of available data
# out:  "plain", "md" (markdown) or "data"

# n_in_dir: Counts data in directories
n_in_dir = function(x){
  data_type = x
  
  # Setup
  dir0 = paste0("Data/", x)
  fs = list.files(dir0, pattern = ".csv", full.names = TRUE)
  fs0 = list.files(dir0, pattern = ".csv")
  
  # Progress setup
  i = 1
  length0 = length(fs)
  
  # Count loop
  x = foreach(f = fs, .combine = "bind_rows") %do% {
    
    f0_i = fs0[i]
    
    # Progress
    cat(i,"of",length0,"::: Reading",f, "                      \r")
    i = i + 1
    
    # Read and count
    suppressWarnings({
      x = read_csv(
        f, show_col_types = FALSE, progress = FALSE
      )
    })
     
    res = x %>% 
      group_by(lang) %>% 
      summarise(
        n = n(),
        n_unique = length(unique(occ1))
      ) %>% 
      arrange(lang) %>% 
      mutate(
        f = f0_i
      )
    
    return(res)
    
  } %>% 
    select(f, lang, n, n_unique)
  
  return(x)
}


unique_in_dir = function(x){
  stop("Not fully implemented. Something is weird in the loop.")
  # Setup
  dir0 = paste0("Data/", x)
  fs = list.files(dir0, pattern = ".csv", full.names = TRUE)
  fs0 = list.files(dir0, pattern = ".csv")
  
  # Progress setup
  i = 1
  length0 = length(fs)
  
  # Count loop
  x = foreach(f = fs, .combine = "bind_rows") %do% {
    
    # Progress
    cat(i,"of",length0,"::: Lang",f, "                      \r")
    i = i + 1
    
    # Read and count
    suppressWarnings({
      x = read_csv(
        f, show_col_types = FALSE, progress = FALSE, n_max = 100
      )
    })
    
    unique_occs = x$occ1
    
    data.frame(occ1 = unique_occs, f = fs0[i])
    
  } %>% 
    select(f, occ1) %>% 
  return(x)
}

million = function(x){
  return(round(x / 1000000, 3))
}

Data_summary = function(out = "plain"){
  # Load counts 
  train = n_in_dir("Training_data") %>% 
    mutate(f = gsub("train", "[x]", f)) %>% 
    rename(
      n_train = n,
      n_train_unique = n_unique
    )
  val1 = n_in_dir("Validation_data1") %>% 
    mutate(f = gsub("val1", "[x]", f)) %>% 
    rename(
      n_val1 = n,
      n_val1_unique = n_unique
    )
  val2 = n_in_dir("Validation_data2") %>% 
    mutate(f = gsub("val2", "[x]", f)) %>% 
    rename(
      n_val2 = n,
      n_val2_unique = n_unique
    )
  test = n_in_dir("Test_data") %>% 
    mutate(f = gsub("test", "[x]", f)) %>% 
    rename(
      n_test = n,
      n_test_unique = n_unique
    )
  
  # Clean counts
  res = train %>% 
    left_join(val1, by = c("f", "lang")) %>% 
    left_join(val2, by = c("f", "lang")) %>% 
    left_join(test, by = c("f", 'lang'))
  
  res = res %>% 
    mutate(
      n = n_train + n_val1 + n_val1 + n_test
    ) %>% 
    arrange(-n) %>% 
    mutate(
      pct_train = paste0(round(100*n/sum(n), 3),"%")
    ) %>% 
    select(-n_val1_unique, -n_val2_unique, -n_test_unique)
  
  Ntrain = res$n_train %>% sum() %>% million()
  Nval1 = res$n_val1 %>% sum() %>% million()
  Nval2 = res$n_val2 %>% sum() %>% million()
  Ntest = res$n_val %>% sum() %>% million()
  capN = res$n %>% sum() %>% million()
  
  # Language summary stats 
  res_lang = res %>% 
    group_by(lang) %>% 
    summarise(
      n_train = sum(n_train),
      n_val1 = sum(n_val1),
      n_val2 = sum(n_val2),
      n_test = sum(n_test),
      n = sum(n)
    ) %>% 
    ungroup() %>% 
    mutate(
      pct = paste0(round(n/sum(n)*100, 3), "%")
    ) %>% 
    arrange(-n)
  
  # Compute summary again...
  res = res %>% 
    group_by(f) %>% 
    summarise(
      n_train = sum(n_train),
      n_val1 = sum(n_val1),
      n_val2 = sum(n_val2),
      n_test = sum(n_test),
      n = sum(n)
    ) %>% 
    arrange(-n) %>% 
    mutate(
      pct_train = paste0(round(100*n/sum(n), 3),"%")
    )
  
  # Load descriptions and citations 
  desc = read_csv2("Data/Summary_data/Data_sources.csv")
  res = res %>% left_join(desc, by = "f")
  
  if(any(is.na(res$Description))){
    warning("Missing description in 'Data/Summary_data/Data_sources.csv'")
  }
  
  res_type = res %>% 
    group_by(Type) %>% 
    summarise(
      n_train = sum(n_train),
      n_val1 = sum(n_val1),
      n_val2 = sum(n_val2),
      n_test = sum(n_test),
      n = sum(n)
    ) %>% 
    ungroup() %>% 
    mutate(
      pct = paste0(round(n/sum(n)*100, 3), "%")
    ) %>% 
    arrange(-n)
  
  # Get results
  if("plain" %in% out){
    cat("\n")
    cat("\nTraining data:   ", Ntrain, "million observations")
    cat("\nValidation data 1: ", Nval1, "million observations")
    cat("\nValidation data 2: ", Nval2, "million observations")
    cat("\nTest data:       ", Ntest, "million observations")
    cat("\n---> In total:   ", capN, "million observations")
    cat("\n\nAmount of data by source:\n")
    knitr::kable(res, "pipe") %>% print()
    res %>% 
      select(f, n, pct_train, lang, Source) %>% 
      knitr::kable("latex", booktabs =TRUE) %>% print()
    knitr::kable(res_lang, "pipe") %>% print()
    knitr::kable(res_type, "pipe") %>% print()
    
    cat("\n\nFor readme:")
    res %>% 
      select(f, n, n_train, Description, Source , lang, Type) %>% 
      rename(
        `File name` = f,
        `N` = n,
        `N train` = n_train,
        Reference = Source,
        Language = lang
      ) %>% 
      knitr::kable("pipe") %>% 
      print()
  }
  
  if("md" %in% out){
    knitr::kable(res, "pipe") %>% print()
    knitr::kable(res_lang, "pipe") %>% print()
    knitr::kable(res_type, "pipe") %>% print()
  }
  
  if("data" %in% out){
    return(list(
      res,
      res_lang,
      res_type
    ))
  }
}


# ==== summary_other_data ====

summary_other_data = function(){
  # Load counts 
  train = n_in_dir("Training_data_other") %>% 
    mutate(f = gsub("train", "[x]", f)) %>% 
    rename(
      n_train = n,
      n_train_unique = n_unique
    )
  val1 = n_in_dir("Validation_data1_other") %>% 
    mutate(f = gsub("val1", "[x]", f)) %>% 
    rename(
      n_val1 = n,
      n_val1_unique = n_unique
    )
  val2 = n_in_dir("Validation_data2_other") %>% 
    mutate(f = gsub("val2", "[x]", f)) %>% 
    rename(
      n_val2 = n,
      n_val2_unique = n_unique
    )
  test = n_in_dir("Test_data_other") %>% 
    mutate(f = gsub("test", "[x]", f)) %>% 
    rename(
      n_test = n,
      n_test_unique = n_unique
    )
  
  
}


# ==== check_if_same() + check_all_in_dir() ====
# check_if_same(): Checks if fname is the same file across both dirs
# check_all_in_dir(): Runs the above function on all files in a dir

check_if_same = function(fname, domain = "Training_data"){
  
  dir1 = paste0("Data/Data_backup/", domain)
  dir2 = paste0("Data/", domain)
  
  fpath1 = paste0(dir1, "/", fname)
  fpath2 = paste0(dir2, "/", fname)
  
  suppressWarnings({
    data1 = read_csv(fpath1, progress = FALSE, show_col_types = FALSE)
    data2 = read_csv(fpath2, progress = FALSE, show_col_types = FALSE)
  })
  
  res = all(data1$RowID == data2$RowID)
  
  return(res)
}

check_all_in_dir = function(dir, verbose = FALSE){
  require(foreach)
  require(tidyverse)
  require(progress)
  
  
  dir1 = paste0("Data/Data_backup/", dir)
  dir2 = paste0("Data/", dir)
  
  # Check that files are the same
  fs1 = list.files(dir1)
  fs2 = list.files(dir2)
  
  test1 = all(fs1 == fs2)
  
  # Setup of progress bar:
  progress_bar_format = paste0(
    "Checking '",
    dir, "': [:bar] :elapsedfull -- :current of :total"
  )
  pb = progress_bar$new(
    total = length(fs1),
    format = progress_bar_format
  )
  
  # Loop that runs test on all files in the two dirs:
  res = foreach(f = fs1, .combine = "c") %do% {
    if(verbose) cat("\n",f)
    res = check_if_same(f, dir)
    pb$tick()
    res
  }
  
  res = all(res)
  return(res)
}
