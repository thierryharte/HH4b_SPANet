"""Script with configurations for each of the datasets that are to be tested for efficiency.

There are two dictionaries; one is a dictionary showing the actual datasets and the other is a list of true data, to which compare the predictions.
"""

import logging

# logger = logging.getLogger(__name__)
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s | %(levelname)s | %(funcName)s | %(message)s",
#     datefmt="%d-%b-%y %H-%M-%S",
# )
logger = logging.getLogger(__name__)


# uncomment the configurations that you want to use
spanet_dir_kevin = "/eos/user/k/kehrler/semester_project/HH4b_SPANet/predictions/"

spanet_dict = {
    "classification_trial_kevin_model_#1": {
        "file": f"{spanet_dir_kevin}hh4b_classification_5jetinput_AN23_183_variables/hh4b_classification_5jetinput_AN23_183_variables.h5",
        "true": "classification_trial_model_#1",
        "label": "hh4b_classification_5jetinput_AN23_183_variables",
        "color": "pink",
    },
#        "classification_trial_kevin_model_#2": {
#        "file": f"{spanet_dir_kevin}hh4b_classification_5jetinput_AN23_183_incomplete_variables/hh4b_classification_5jetinput_AN23_183_incomplete_variables.h5",
#        "true": "classification_trial_model_#2",
#        "label": "hh4b_classification_5jetinput_AN23_183_incomplete_variables",
#        "color": "blue",
#    },
}

true_path_thierry = "/eos/user/t/tharte/Analysis_data/spanet_samples/classification/"
true_dict = {
    "classification_trial_model_#1": {
        "name": f"{true_path_thierry}1_15_9_jetgoodfromhiggsordered5jets_additional_parameters/columns_for_sig_bkg_classifierJetGoodFromHiggsOrdered5Jets_test.h5"
    },
#    "classification_trial_model_#2": {
#        "name": f"{true_path_thierry}1_15_8_jetgoodfromhiggsordered5jets/columns_for_classifierJetGoodFromHiggsOrdered5Jets_test.h5"
#    },
}