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
  fs = list.files(dir)
  fs = fs[grepl(".csv", fs)]
  foreach(f = fs, .combine = "bind_rows") %do% {
    read_csv(paste0(dir,"/",f), guess_max = 100000) %>% mutate(file = f)
  }
}
