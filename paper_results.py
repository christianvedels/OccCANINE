from Project_dissemination.Paper_replication_package.Model_eval_python.predict_ood import main as predict_ood_main
from Project_dissemination.Paper_replication_package.Model_eval_python.finetune import main as finetune_main
from Project_dissemination.Paper_replication_package.Model_eval_python.agreement_german_sources import main as agreement_german_sources_main
from Project_dissemination.Paper_replication_package.Model_eval_python.predict_test import main as predict_test_main
from Project_dissemination.Paper_replication_package.Model_eval_python.embeddings import main as embeddings_main
from Project_dissemination.Paper_replication_package.Model_eval_python.other_systems_eval import main as other_systems_eval_main
from Project_dissemination.Paper_replication_package.Model_eval_python.threshold_tuning import wrapper

if __name__ == "__main__":
    # Runs all the prerequisite scripts for the paper results

    # predict_test_main()
    # agreement_german_sources_main()
    predict_ood_main()
    # finetune_main()
    # embeddings_main()
    # other_systems_eval_main()

