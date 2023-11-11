# Functions
# Created:  2023-05-23
# Authors:  Christian Vedel [christian-vs@sam.sdu.dk],
#
# Purpose:  Contains all the functions used in data cleaning


# ==== sub_scandi ====
# Replaces all Scandinavian letters with their standard English alphabet 
# counterpart
# x: Character to clean

sub_scandi = function(x){
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
Keep_only_relevant = function(x){
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
    "split"
  )
  
  res = x[,which(names(x)%in%relevant_vars)]
  return(res)
}

# ==== write_csv0 ====
# write_csv, but with a message
write_csv0 = function(x, fname){
  cat("\nSaving", fname, "with", NROW(x)/1000,"thousands rows")
  write_csv(x, fname)
}

# ==== Save_train_val_test ====
# Saves train test val data
# x:        Data containing 'split'
# Name      Name of the data. Will be saved as [Name]_[x].csv, where x is test, train, etc.
# language: Language 'da', 'en', 'nl' or 'se', ...

Save_train_val_test = function(x, Name, language = NA){
  # Throw error if incorrect language
  valid_languages = c('da', 'en', 'nl', 'se', 'no', 'fr', 'ca', 'unk', 'de', 'is', 'unk', 'it')
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
      RowID = paste0(Name, RowID),
      lang = language
    )
  
  # Add na year if it is missing
  if((!"Year" %in% names(x))){
    x = x %>% 
      mutate(Year = NA)
  }
  
  # Filter data
  x_test = x %>% 
    filter(split == "Test")
  x_val = x %>% 
    filter(grepl("Val", split))
  x_train = x %>% 
    filter(split == "Train")
  
  # Make file names
  fname_test = paste0("Data/Test_data/", Name, "_test.csv")
  fname_val = paste0("Data/Validation_data/", Name, "_val.csv")
  fname_train = paste0("Data/Training_data/", Name, "_train.csv")
  
  x_test %>% 
    write_csv0(fname_test)
  x_val %>% 
    write_csv0(fname_val)
  x_train %>% 
    write_csv0(fname_train)
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
    cat(i,"of",length0,"::: Reading",f, "                      \r")
    i = i + 1
    
    # Read and count
    suppressWarnings({
      x = read_csv(
        f, show_col_types = FALSE, progress = FALSE
      )
      n = x %>% NROW() # Count number of rows  
      n_unique = x %>% distinct(occ1) %>% NROW()
        
      data.frame(n = n, n_unique = n_unique)
    })
    
    
  } %>% mutate(f = fs0) %>% 
    select(f, n, n_unique)
  return(x)
}

lang_in_dir = function(x){
  
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
    
    lang = x$lang %>% unique()
    if(length(lang)>1){
      stop("More than one language in: ", f)
    }
    data.frame(lang = lang)
    
  } %>% mutate(f = fs0) %>% 
    select(f, lang)
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
  val = n_in_dir("Validation_data") %>% 
    mutate(f = gsub("val", "[x]", f)) %>% 
    rename(
      n_val = n,
      n_val_unique = n_unique
    )
  test = n_in_dir("Test_data") %>% 
    mutate(f = gsub("test", "[x]", f)) %>% 
    rename(
      n_test = n,
      n_test_unique = n_unique
    )
  
  # Clean counts
  res = train %>% 
    left_join(val, by = "f") %>% 
    left_join(test, by = "f")
  
  res = res %>% 
    mutate(
      n = n_train + n_val + n_test
    ) %>% 
    arrange(-n) %>% 
    mutate(
      pct_train = paste0(round(100*n/sum(n), 3),"%"),
      pct_train_unique = paste0(round(100*n_train_unique/sum(n_train_unique), 3),"%")
    ) %>% 
    select(-n_val_unique, -n_test_unique)
  
  Ntrain = res$n_train %>% sum() %>% million()
  Nval = res$n_val %>% sum() %>% million()
  Ntest = res$n_val %>% sum() %>% million()
  capN = res$n %>% sum() %>% million()
  
  # Identify languages
  lang_train = lang_in_dir("Training_data") %>% 
    mutate(f = gsub("train", "[x]", f)) %>% 
    rename(lang_train = lang)
  lang_val = lang_in_dir("Validation_data") %>% 
    mutate(f = gsub("val", "[x]", f)) %>% 
    rename(lang_val = lang)
  lang_test = lang_in_dir("Test_data") %>% 
    mutate(f = gsub("test", "[x]", f)) %>% 
    rename(lang_test = lang)
  
  langs = lang_train %>% 
    left_join(lang_val, by = "f") %>% 
    left_join(lang_test, by = "f")
  
  # test similarity of languages 
  test1 = !all(langs$lang_train==langs$lang_train)
  test2 = !all(langs$lang_train==langs$lang_val)
  if(test1 | test2){
    stop("Languages are not the same somehow")
  }
  
  langs = langs %>% 
    select(f, lang_train) %>% 
    rename(lang = lang_train)
  
  res = res %>% 
    left_join(langs, by = "f")
  
  # Language summary stats 
  lang_sum = res %>% 
    group_by(lang) %>% 
    summarise(
      n_train = sum(n_train),
      n_val = sum(n_val),
      n_test = sum(n_test),
      n = sum(n)
    ) %>% 
    ungroup() %>% 
    mutate(
      pct = paste0(round(n/sum(n)*100, 3), "%")
    ) %>% 
    arrange(-n)
  
  # Load descriptions and citations 
  desc = read_csv2("Data/Summary_data/Data_sources.csv")
  desc %>% select(-lang)
  res = res %>% left_join(desc)
  
  if(any(is.na(res$Description))){
    warning("Missing description in 'Data/Summary_data/Data_sources.csv'")
  }
  
  res_type = res %>% 
    group_by(Type) %>% 
    summarise(
      n_train = sum(n_train),
      n_val = sum(n_val),
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
    cat("\nValidation data: ", Nval, "million observations")
    cat("\nTest data:       ", Ntest, "million observations")
    cat("\n---> In total:   ", capN, "million observations")
    cat("\n\nAmount of data by source:\n")
    knitr::kable(res, "pipe") %>% print()
    knitr::kable(lang_sum, "pipe") %>% print()
    knitr::kable(res_type, "pipe") %>% print()
  }
  
  if("md" %in% out){
    knitr::kable(res, "pipe") %>% print()
    knitr::kable(lang_sum, "pipe") %>% print()
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
