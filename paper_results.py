from Project_dissemination.Paper_replication_package.Model_eval_python.threshold_tuning import main as threshold_tuning_main
from Project_dissemination.Paper_replication_package.Model_eval_python.predict_test import main as predict_test_main
from Project_dissemination.Paper_replication_package.Model_eval_python.predict_ood import main as predict_ood_main
from Project_dissemination.Paper_replication_package.Model_eval_python.predict_testbylang import main as predict_testbylang_main
from Project_dissemination.Paper_replication_package.Model_eval_python.predict_testby_source import main as predict_testby_source_main
from Project_dissemination.Paper_replication_package.Model_eval_python.embeddings import main as embeddings_main
from Project_dissemination.Paper_replication_package.Model_eval_python.other_systems_eval import main as other_systems_eval_main
from Project_dissemination.Paper_replication_package.Model_eval_python.agreement_german_sources import main as agreement_german_sources_main

if __name__ == "__main__":

    tr = True  # Set to True for toy run, False for full run
    # Runs all the prerequisite scripts for the paper results
    threshold_tuning_main(toyrun=tr)
    predict_test_main(toyrun=tr)
    predict_ood_main()
    predict_testbylang_main(toyrun=tr)
    predict_testby_source_main(toyrun=tr)
    embeddings_main(toyrun=tr)
    other_systems_eval_main(toyrun=tr)
    agreement_german_sources_main()

