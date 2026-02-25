from Project_dissemination.Paper_replication_package.Model_eval_python.threshold_tuning import main as threshold_tuning_main
from Project_dissemination.Paper_replication_package.Model_eval_python.predict_test import main as predict_test_main
from Project_dissemination.Paper_replication_package.Model_eval_python.predict_ood import main as predict_ood_main
from Project_dissemination.Paper_replication_package.Model_eval_python.predict_testbylang import main as predict_testbylang_main
from Project_dissemination.Paper_replication_package.Model_eval_python.predict_testby_source import main as predict_testby_source_main
from Project_dissemination.Paper_replication_package.Model_eval_python.embeddings import main as embeddings_main
from Project_dissemination.Paper_replication_package.Model_eval_python.other_systems_eval import main as other_systems_eval_main
from Project_dissemination.Paper_replication_package.Model_eval_python.agreement_german_sources import main as agreement_german_sources_main
from Project_dissemination.Paper_replication_package.Model_eval_python.topk_testing_ood import main as topk_testing_main
from Project_dissemination.Paper_replication_package.Model_eval_python.topk_test import main as topk_test_main
from Project_dissemination.Paper_replication_package.Model_eval_python.performance_benchmark import main as performance_benchmark_main
from Project_dissemination.Paper_replication_package.Model_eval_python.predict_test_flat import main as predict_test_flat_main

if __name__ == "__main__":
    tr = False  # Set to True for toy run, False for full run
    DATA_PATH = r"Data" # Overall path to all data files for OccCANINE 

    # Runs all the prerequisite scripts for the paper results
    # threshold_tuning_main(toyrun=tr, data_path=f"{DATA_PATH}/Validation_data1/*.csv")
    # predict_test_main(toyrun=tr, data_path=f"{DATA_PATH}/Test_data/*.csv")
    # predict_test_main(toyrun=tr, data_path=f"{DATA_PATH}/Test_data_unique_strings/*.csv", name="test_unique") # Same as predct_test but for unique strings only [1]
    # topk_test_main(toyrun=tr, data_path=f"{DATA_PATH}/Test_data/*.csv", name="test", K=10, fix_duplicate_id=True)
    # topk_test_main(toyrun=tr, data_path=f"{DATA_PATH}/Test_data_unique_strings/*.csv", name="test_unique", K=10, fix_duplicate_id=True)  # Same as topk_test but for unique strings only [1]
    # predict_ood_main(data_path=f"{DATA_PATH}/OOD_data")
    # topk_testing_main(data_path=f"{DATA_PATH}/OOD_data", K=10, toyrun=tr)
    # predict_testbylang_main(toyrun=tr, data_path=f"{DATA_PATH}/Test_data/*.csv")
    # predict_testby_source_main(toyrun=tr, data_path=f"{DATA_PATH}/Test_data/*.csv")
    # embeddings_main(toyrun=tr, data_path=f"{DATA_PATH}/Test_data/*.csv")
    # other_systems_eval_main(toyrun=tr, data_path=f"{DATA_PATH}/Test_data_other", mod_path=f"{DATA_PATH}/models")
    # agreement_german_sources_main(data_path=f"{DATA_PATH}/OOD_data")
    # performance_benchmark_main(n_obs=10000, data_path=f"{DATA_PATH}/Test_data/*.csv", behavior="fast")
    # performance_benchmark_main(n_obs=10000, data_path=f"{DATA_PATH}/Test_data/*.csv", behavior="good")
    predict_test_flat_main(toyrun=tr, data_path=f"{DATA_PATH}/Test_data/*.csv", thresholds= [round(t * 0.01, 2) for t in range(1, 100)], name="test_flat")

# Note:
# [1] The "test_unique" run evaluation on occs not found in training. E.g. "baker" is a common occupation and occurs multiple times.
#     This evaluation only includes one occurrence of strings where such cases have been sorted away. 
