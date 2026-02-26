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
    "classification_trial_kevin": {
        "file": f"{spanet_dir_kevin}example_model_from_thierry/example_model_from_thierry.h5",
        "true": "classification_trial",
        "label": "First test 4jet inputs probably bad model",
        "color": "pink",
    },
}

true_path_thierry = "/eos/user/t/tharte/Analysis_data/spanet_samples/classification/"
true_dict = {
    "classification_trial": {
        "name": f"{true_path_thierry}conversion_with_vbf/columns_for_classifier_test.h5"
    },
}
