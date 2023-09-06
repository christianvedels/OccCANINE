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
  scandi_letters = c("Æ",
                     "æ",
                     "Ø",
                     "ø",
                     "Å",
                     "å",
                     "ö",
                     "Ö",
                     "ä",
                     "Ä")
  
  replacement = c("Ae",
                  "ae",
                  "Oe",
                  "oe",
                  "Aa",
                  "aa",
                  "oe",
                  "Oe",
                  "ae",
                  "Ae"
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
  valid_languages = c('da', 'en', 'nl', 'se', 'no')
  if(language %in% valid_languages){
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
