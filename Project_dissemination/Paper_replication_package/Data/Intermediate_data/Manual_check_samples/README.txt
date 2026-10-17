Manual check sample of unique OOD strings (DWI and UK Bankruptcies)
Created by Model_eval_R/005_OOD_performance.R on 2026-10-17 
R version 4.4.1 (2024-06-14 ucrt) 

Unique strings sampled per source (simple random sample): 500 
Seed: 20 (set before each source's draw)

Input files (md5):
  predictions_EN_dwi.csv  62edb8135cd2356f2327df66582911af
  predictions_EN_Bankruptcy_KornLacroix.csv  d1b74e219d68835242d58b1f92252e9d 

Sampling frame (unique strings, 'nan' / empty occupations excluded):
          source n_unique_strings
             DWI             2348
 UK Bankruptcies             4948

Fill in correct_hisco_1..5 (use -1 for no occupation) and comment,
and save as OOD_manual_check_labelled.csv (';'-separated) in this folder.
