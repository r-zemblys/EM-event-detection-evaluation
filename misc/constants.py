UU_FILTER = [(100, 255), (255, 100)]  # Unmatched undefined filter
MU_FILTER = [(100, 100)]  # matched undefined filter
UNC_FILTER = [(0, -1), (-1, 0)]  # unmatched negative class filter
UNC_MAPPING = {"tn": 0, "error": 1}  # unmatched negative class mapping

event_mapping = {
    0: {"label": "Negative", "color": "#d3d3d3"},
    1: {"label": "Fixation", "color": "#3b5b92"},  # and positive
    2: {"label": "Saccade", "color": "#d9544d"},
    3: {"label": "PSO", "color": "#39ad48"},
    4: {"label": "Pursuit", "color": "#ff8c00"},
    5: {"label": "Blink", "color": "#8b008b"},
    11: {"label": "OKN"},
    12: {"label": "VOR"},
    13: {"label": "OKN+VOR"},
    14: {"label": "Head pursuit"},
    100: {"label": "Undefined", "color": "#708090"},
    101: {"label": "negative", "color": "#d3d3d3"},
    255: {"label": "Reserved"},
}

matcher_map = {
    "nld-sample": "Sample\nlevel NLD",
    "nld-event": "Event\nlevel NLD",
    "sample": "Sample",
    # "plurality-voting": "Plurality\nvoting",
    "majority-voting": "Majority\nvoting",
    "earliest-overlap": "Earliest\noverlap",
    "overlap/one-match": "Overlap",
    "overlap": "Overlap (all)",
    "maximum-overlap": "Maximum\noverlap",
    "iou": "Maximum\nIoU",
    "iou/05": "IoU>0.5",
}

metric_map = {
    "nld": r"1$-NLD$",
    "_nld": "NLD",
    "accuracy": "Accuracy",
    "accuracy_balanced": "Balanced accuracy",
    "precision": "Precision",
    "sensitivity": "Sensitivity",
    "specificity": "Specificity",
    "f1_score": "F1-score",
    # "auc": "AUC",
    "iou": "JI",
    "kappa": "$\kappa$",
    "pearsonr": "Pearson`s r",
    "spearmanr": "Spearman`s rho",
    "kendalltau": "Kendall`s tau",
    "mcc": "$MCC$",
}

alg_map = {
    "Hollywood2EM_berg": "Berg et al.",
    "Hollywood2EM_blstm": "BLSTM",
    "Hollywood2EM_dorr": "Dorr et al.",
    "Hollywood2EM_handlabeller_1": "Student coder",
    "Hollywood2EM_idt": "I-DT",
    "Hollywood2EM_ihmm": "I-HMM",
    "Hollywood2EM_ikf": "I-KF",
    "Hollywood2EM_imst": "I-MST",
    "Hollywood2EM_ivdt": "I-VDT",
    "Hollywood2EM_ivmp": "I-VMP",
    "Hollywood2EM_ivvt": "I-VVT",
    "Hollywood2EM_ivt": "I-VT",
    "Hollywood2EM_larsson": "Larsson et al.",
    "Hollywood2EM_remodnav": "REMoDNaV",
    "Hollywood2EM_sp_tool": "sp_tool",
    "Hollywood2EM_sp_tool_smoothed": "sp_tool_smoothed",
}

exclude_map = {
    "job_gazecom": {
        "pr": [
            "GazeCom_berg-add",
            "GazeCom_handlabeller1",
            "GazeCom_handlabeller2",
        ]
    },
    "job_hollywood2": {
        "pr": [
            "Hollywood2EM_handlabeller_1",
        ],
        "fname": [
            "Hollywood2EM_handlabeller_final/train/",
        ],
    },
}

plot_kwargs = {
    "kind": "point",
    "ci": "sd",
    "legend": False,
    "dodge": 0.25,
    "linestyles": "-",
    "markers": "o",
    "errwidth": 0.5,
}

kwargs_legend = {
    "frameon": False,
    "fontsize": "small",
    "handletextpad": 0.1,
    "columnspacing": 0.1,
    "loc": "upper left",
    "bbox_to_anchor": (-0.05, 1.1),
    "ncol": 6,
}
