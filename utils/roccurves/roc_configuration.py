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
#    "spanet_dict_example_model_from_thierry": {
#        "true": "true_dict_example_model_from_thierry",
#        "file": f"{spanet_dir_kevin}example_model_from_thierry/example_model_from_thierry.h5",
#        "label": "example_model_from_thierry",
#        "color": "pink",
#    },

#     "spanet_dict_hh4b_classification_trial": {
#         "file": f"{spanet_dir_kevin}hh4b_classification_trial/hh4b_classification_trial.h5",
#         "label": "hh4b_classification_trial",
#         "true": "true_dict_hh4b_classification_trial",
#         "color": "pink",
#    },

#     "spanet_dict_hh4b_classification_5jetinput": {
#        "file": f"{spanet_dir_kevin}hh4b_classification_5jetinput/hh4b_classification_5jetinput.h5",
#        "true": "true_dict_hh4b_classification_5jetinput",
#        "label": "hh4b_classification_5jetinput",
#        "color": "pink",
#    },

#    "spanet_dict_hh4b_classification_btagPNetB_5wp": {
#        "file": f"{spanet_dir_kevin}hh4b_classification_btagPNetB_5wp/hh4b_classification_btagPNetB_5wp.h5",
#        "true": "true_dict_hh4b_classification_btagPNetB_5wp",
#        "label": "hh4b_classification_btagPNetB_5wp",
#        "color": "pink",
#    },

    "spanet_dict_hh4b_classification_AN23_184_incomplete": {
       "file": f"{spanet_dir_kevin}hh4b_classification_AN23_184_incomplete/hh4b_classification_AN23_184_incomplete.h5",
       "true": "true_dict_hh4b_classification_AN23_184_incomplete",
       "label": "hh4b_classification_AN23_184_incomplete",
       "color": "pink",
   },

#    "spanet_dict_hh4b_classification_AN23_184": {
#        "file": f"{spanet_dir_kevin}hh4b_classification_AN23_184/hh4b_classification_AN23_184.h5",
#        "true": "true_dict_hh4b_classification_AN23_184",
#        "label": "hh4b_classification_AN23_184",
#        "color": "pink",
#    },

#    "spanet_dict_hh4b_classification_var000000": {
#        "file": f"{spanet_dir_kevin}hh4b_classification_var000000/hh4b_classification_var000000.h5",
#        "true": "true_dict_hh4b_classification_var0000##",
#        "label": "hh4b_classification_var000000",
#        "color": "pink",
#    },

#    "spanet_dict_hh4b_classification_var000001": {
#        "file": f"{spanet_dir_kevin}hh4b_classification_var000001/hh4b_classification_var000001.h5",
#        "true": "true_dict_hh4b_classification_var0000##",
#        "label": "hh4b_classification_var000001",
#        "color": "pink",
#    },

#    "spanet_dict_hh4b_classification_var000002": {
#        "file": f"{spanet_dir_kevin}hh4b_classification_var000002/hh4b_classification_var000002.h5",
#        "true": "true_dict_hh4b_classification_var0000##",
#        "label": "hh4b_classification_var000002",
#        "color": "pink",
#    },

#    "spanet_dict_hh4b_classification_var000003": {
#        "file": f"{spanet_dir_kevin}hh4b_classification_var000003/hh4b_classification_var000003.h5",
#        "true": "true_dict_hh4b_classification_var0000##",
#        "label": "hh4b_classification_var000003",
#        "color": "pink",
#    },

#    "spanet_dict_hh4b_classification_var000004": {
#        "file": f"{spanet_dir_kevin}hh4b_classification_var000004/hh4b_classification_var000004.h5",
#        "true": "true_dict_hh4b_classification_var0000##",
#        "label": "hh4b_classification_var000004",
#        "color": "pink",
#    },

#    "spanet_dict_hh4b_classification_var000005": {
#        "file": f"{spanet_dir_kevin}hh4b_classification_var000005/hh4b_classification_var000005.h5",
#        "true": "true_dict_hh4b_classification_var0000##",
#        "label": "hh4b_classification_var000005",
#        "color": "pink",
#    },

#    "spanet_dict_hh4b_classification_var000006": {
#        "file": f"{spanet_dir_kevin}hh4b_classification_var000006/hh4b_classification_var000006.h5",
#        "true": "true_dict_hh4b_classification_var0000##",
#        "label": "hh4b_classification_var000006",
#        "color": "pink",
#    },

#    "spanet_dict_hh4b_classification_var000007": {
#        "file": f"{spanet_dir_kevin}hh4b_classification_var000007/hh4b_classification_var000007.h5",
#        "true": "true_dict_hh4b_classification_var0000##",
#        "label": "hh4b_classification_var000007",
#        "color": "pink",
#    },

#    "spanet_dict_hh4b_classification_var000008": {
#        "file": f"{spanet_dir_kevin}hh4b_classification_var000008/hh4b_classification_var000008.h5",
#        "true": "true_dict_hh4b_classification_var0000##",
#        "label": "hh4b_classification_var000008",
#        "color": "pink",
#    },

#    "spanet_dict_hh4b_classification_var000009": {
#        "file": f"{spanet_dir_kevin}hh4b_classification_var000009/hh4b_classification_var000009.h5",
#        "true": "true_dict_hh4b_classification_var0000##",
#        "label": "hh4b_classification_var000009",
#        "color": "pink",
#    },

#    "spanet_dict_hh4b_classification_var000010": {
#        "file": f"{spanet_dir_kevin}hh4b_classification_var000010/hh4b_classification_var000010.h5",
#        "true": "true_dict_hh4b_classification_var0000##",
#        "label": "hh4b_classification_var000010",
#        "color": "pink",
#    },

#    "spanet_dict_hh4b_classification_var000011": {
#        "file": f"{spanet_dir_kevin}hh4b_classification_var000011/hh4b_classification_var000011.h5",
#        "true": "true_dict_hh4b_classification_var0000##",
#        "label": "hh4b_classification_var000011",
#        "color": "pink",
#    },

#    "spanet_dict_hh4b_classification_var000012": {
#        "file": f"{spanet_dir_kevin}hh4b_classification_var000012/hh4b_classification_var000012.h5",
#        "true": "true_dict_hh4b_classification_var0000##",
#        "label": "hh4b_classification_var000012",
#        "color": "pink",
#    },

#    "spanet_dict_hh4b_classification_var000013": {
#        "file": f"{spanet_dir_kevin}hh4b_classification_var000013/hh4b_classification_var000013.h5",
#        "true": "true_dict_hh4b_classification_var0000##",
#        "label": "hh4b_classification_var000013",
#        "color": "pink",
#    },

#    "spanet_dict_hh4b_classification_var000014": {
#        "file": f"{spanet_dir_kevin}hh4b_classification_var000014/hh4b_classification_var000014.h5",
#        "true": "true_dict_hh4b_classification_var0000##",
#        "label": "hh4b_classification_var000014",
#        "color": "pink",
#    },

#    "spanet_dict_hh4b_classification_var000015": {
#        "file": f"{spanet_dir_kevin}hh4b_classification_var000015/hh4b_classification_var000015.h5",
#        "true": "true_dict_hh4b_classification_var0000##",
#        "label": "hh4b_classification_var000015",
#        "color": "pink",
#    },

#    "spanet_dict_hh4b_classification_var000016": {
#        "file": f"{spanet_dir_kevin}hh4b_classification_var000016/hh4b_classification_var000016.h5",
#        "true": "true_dict_hh4b_classification_var0000##",
#        "label": "hh4b_classification_var000016",
#        "color": "pink",
#    },

#    "spanet_dict_hh4b_classification_var000017": {
#        "file": f"{spanet_dir_kevin}hh4b_classification_var000017/hh4b_classification_var000017.h5",
#        "true": "true_dict_hh4b_classification_var0000##",
#        "label": "hh4b_classification_var000017",
#        "color": "pink",
#    },

#    "spanet_dict_hh4b_classification_var000018": {
#        "file": f"{spanet_dir_kevin}hh4b_classification_var000018/hh4b_classification_var000018.h5",
#        "true": "true_dict_hh4b_classification_var0000##",
#        "label": "hh4b_classification_var000018",
#        "color": "pink",
#    },

#    "spanet_dict_hh4b_classification_var000019": {
#        "file": f"{spanet_dir_kevin}hh4b_classification_var000019/hh4b_classification_var000019.h5",
#        "true": "true_dict_hh4b_classification_var0000##",
#        "label": "hh4b_classification_var000019",
#        "color": "pink",
#    },

#    "spanet_dict_hh4b_classification_var000020": {
#        "file": f"{spanet_dir_kevin}hh4b_classification_var000020/hh4b_classification_var000020.h5",
#        "true": "true_dict_hh4b_classification_var0000##",
#        "label": "hh4b_classification_var000020",
#        "color": "pink",
#    },

#    "spanet_dict_hh4b_classification_var000021": {
#        "file": f"{spanet_dir_kevin}hh4b_classification_var000021/hh4b_classification_var000021.h5",
#        "true": "true_dict_hh4b_classification_var0000##",
#        "label": "hh4b_classification_var000021",
#        "color": "pink",
#    },
}

true_path_thierry = "/eos/user/t/tharte/Analysis_data/spanet_samples/classification/"
true_dict = {
       "true_dict_example_model_from_thierry": {
       "name": f"{true_path_thierry}conversion_with_vbf/columns_for_classifier_test.h5"
    },

       "true_dict_hh4b_classification_trial": {
       "name": f"{true_path_thierry}conversion_with_vbf/columns_for_classifierJetGoodFromHiggsOrdered_test.h5"
    },

       "true_dict_hh4b_classification_5jetinput": {
       "name": f"{true_path_thierry}1_15_8_jetgoodfromhiggsordered5jets/columns_for_classifierJetGoodFromHiggsOrdered5Jets_test.h5"
    },

        "true_dict_hh4b_classification_btagPNetB_5wp": {
        "name": f"{true_path_thierry}1_15_8_jetgoodfromhiggsordered5jets/columns_for_classifierJetGoodFromHiggsOrdered5Jets_test.h5"
       },

       "true_dict_hh4b_classification_AN23_184_incomplete": {
       "name": f"{true_path_thierry}1_15_8_jetgoodfromhiggsordered5jets/columns_for_classifierJetGoodFromHiggsOrdered5Jets_test.h5"
   },

    "true_dict_hh4b_classification_AN23_184": {
        "name": f"{true_path_thierry}1_15_9_jetgoodfromhiggsordered5jets_additional_parameters/columns_for_sig_bkg_classifierJetGoodFromHiggsOrdered5Jets_test.h5"
    },

    "true_dict_hh4b_classification_var0000##": {
        "name": f"{true_path_thierry}1_15_9_jetgoodfromhiggsordered5jets_additional_parameters/columns_for_sig_bkg_classifierJetGoodFromHiggsOrdered5Jets_test.h5"
    },
}
