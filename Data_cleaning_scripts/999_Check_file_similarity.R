# Check file simiarity
# Updated:    2024-09-03
# Auhtors:    Christian Vedel [christian-vs@sam.sdu.dk],
#
# Purpose:    Check if all Row IDs are identical across different versions of
#             data in the general folder and in the "Data/Data_backup" folder


# Instructions:
# 1.  Take "Training_data/", "Validation_data1/" and "Test_data/" and clip to
#     "Data_backup/"
# 2.  Run "101_Train_test_val_split.R" to regenerate all data.
# 3.  Execute the code below. 

source("Data_cleaning_scripts/000_Functions.R")
test1 = check_all_in_dir("Training_data")
test2 = check_all_in_dir("Validation_data1")
test3 = check_all_in_dir("Test_data")

result = test1*test2*test3 == 1
if(!result) stop("Some RowIDs do not match.")
