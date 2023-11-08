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
