import os
import copy
from operator import itemgetter

import arff
from tqdm import tqdm

import numpy as np
import pandas as pd

try:
    from __init__ import ROOT
except ImportError:
    ROOT = None

from misc import utils

##
TUM_EVENT_MAPPING = {
    0: 0,  # undefined
    1: 1,  # fixation
    2: 2,  # saccade
    3: 4,  # Pursuit
    4: 0,  # Noise
    "UNKNOWN": 0,
    "FIX": 1,
    "SACCADE": 2,
    "PSO": 3,
    "SP": 4,
    "NOISE": 0,
    "BLINK": 5,
    "NOISE_CLUSTER": 0,
    "unassigned": 0,
    "fixation": 1,
    "saccade": 2,
    "noise": 0,
    "OKN": 11,
    "VOR": 12,
    "OKN+VOR": 13,
    "head_pursuit": 14,
}

alg_mapping_gazecom = {
    "berg": {"dir": "output_berg_itti", "col": "itti", "suffix": None},
    "larsson": {"dir": "output_larsson", "col": "larsson", "suffix": "larsson"},
    "sp_tool": {"dir": "output_sp_tool", "col": "EYE_MOVEMENT_TYPE", "suffix": None},
    "agtzidis-add": {
        "dir": "algorithms_to_compare/Agtzidis et al. - EYE_MOVEMENT_TYPE",
        "col": "EYE_MOVEMENT_TYPE",
        "suffix": None,
    },
    "berg-add": {
        "dir": "algorithms_to_compare/Berg et al. - itti",
        "col": "itti",
        "suffix": None,
    },
    "dorr-add": {
        "dir": "algorithms_to_compare/Dorr et al. - dsf",
        "col": "dsf",
        "suffix": None,
    },
    "ivdt-optim-add": {
        "dir": "algorithms_to_compare/I-VDT (optimised) - komogortsev_ivdt",
        "col": "komogortsev_ivdt",
        "suffix": None,
    },
    "ivmp-optim-add": {
        "dir": "algorithms_to_compare/I-VMP (optimised) - komogortsev_ivmp",
        "col": "komogortsev_ivmp",
        "suffix": None,
    },
    "ivvt-optim-add": {
        "dir": "algorithms_to_compare/I-VVT (optimised) - komogortsev_ivvt",
        "col": "komogortsev_ivvt",
        "suffix": None,
    },
    "ivvt-add": {
        "dir": "algorithms_to_compare/Komogortsev - komogortsev_ivt, komogortsev_idt, komogortsev_ihmm, komogortsev_ikf, komogortsev_imst",
        "col": "komogortsev_ivvt",
        "suffix": None,
    },
    "ivdt-add": {
        "dir": "algorithms_to_compare/Komogortsev - komogortsev_ivt, komogortsev_idt, komogortsev_ihmm, komogortsev_ikf, komogortsev_imst",
        "col": "komogortsev_ivdt",
        "suffix": None,
    },
    "ivmp-add": {
        "dir": "algorithms_to_compare/Komogortsev - komogortsev_ivt, komogortsev_idt, komogortsev_ihmm, komogortsev_ikf, komogortsev_imst",
        "col": "komogortsev_ivmp",
        "suffix": None,
    },
    "ivt-add": {
        "dir": "algorithms_to_compare/Komogortsev - komogortsev_ivt, komogortsev_idt, komogortsev_ihmm, komogortsev_ikf, komogortsev_imst",
        "col": "komogortsev_ivt",
        "suffix": None,
    },
    "idt-add": {
        "dir": "algorithms_to_compare/Komogortsev - komogortsev_ivt, komogortsev_idt, komogortsev_ihmm, komogortsev_ikf, komogortsev_imst",
        "col": "komogortsev_idt",
        "suffix": None,
    },
    "ihmm-add": {
        "dir": "algorithms_to_compare/Komogortsev - komogortsev_ivt, komogortsev_idt, komogortsev_ihmm, komogortsev_ikf, komogortsev_imst",
        "col": "komogortsev_ihmm",
        "suffix": None,
    },
    "ikf-add": {
        "dir": "algorithms_to_compare/Komogortsev - komogortsev_ivt, komogortsev_idt, komogortsev_ihmm, komogortsev_ikf, komogortsev_imst",
        "col": "komogortsev_ikf",
        "suffix": None,
    },
    "imst-add": {
        "dir": "algorithms_to_compare/Komogortsev - komogortsev_ivt, komogortsev_idt, komogortsev_ihmm, komogortsev_ikf, komogortsev_imst",
        "col": "komogortsev_imst",
        "suffix": None,
    },
    "larsson-add": {
        "dir": "algorithms_to_compare/Larsson et al. - larsson",
        "col": "larsson",
        "suffix": "larsson",
    },
    "startsev-add": {
        "dir": "algorithms_to_compare/Startsev et al. (updated) - EYE_MOVEMENT_TYPE",
        "col": "EYE_MOVEMENT_TYPE",
        "suffix": None,
    },
}

alg_mapping_hollywood2_em = {
    "berg": {"dir": "output_berg", "col": "berg", "suffix": None},
    "blstm": {"dir": "output_blstm", "col": "EYE_MOVEMENT_TYPE", "suffix": None},
    "dorr": {"dir": "output_dorr", "col": "dorr", "suffix": None},
    "ivvt": {"dir": "output_komogortsev", "col": "komogortsev_ivvt", "suffix": None},
    "ivdt": {"dir": "output_komogortsev", "col": "komogortsev_ivdt", "suffix": None},
    "ivmp": {"dir": "output_komogortsev", "col": "komogortsev_ivmp", "suffix": None},
    "ivt": {"dir": "output_komogortsev", "col": "komogortsev_ivt", "suffix": None},
    "ikf": {"dir": "output_komogortsev", "col": "komogortsev_ikf", "suffix": None},
    "idt": {"dir": "output_komogortsev", "col": "komogortsev_idt", "suffix": None},
    "imst": {"dir": "output_komogortsev", "col": "komogortsev_imst", "suffix": None},
    "ihmm": {"dir": "output_komogortsev", "col": "komogortsev_ihmm", "suffix": None},
    "larsson": {"dir": "output_larsson", "col": "larsson", "suffix": "larsson"},
    "remodnav": {"dir": "output_remodnav", "col": "EYE_MOVEMENT_TYPE", "suffix": None},
    "sp_tool": {"dir": "output_sp_tool", "col": "EYE_MOVEMENT_TYPE", "suffix": None},
    "sp_tool_smoothed": {
        "dir": "output_sp_tool_smoothed",
        "col": "EYE_MOVEMENT_TYPE",
        "suffix": None,
    },
}

alg_mapping_360em = {
    "is5t_combined": {
        "dir": "output_I-S5T_combined",
        "col": "primary_label",
        "suffix": None,
    },
    "is5t_eh": {"dir": "output_I-S5T_E+H", "col": "primary_label", "suffix": None},
    "is5t_fov": {"dir": "output_I-S5T_FOV", "col": "primary_label", "suffix": None},
}

alg_mapping_360em_secondary = {
    "is5t_combined": {
        "dir": "output_I-S5T_combined",
        "col": "secondary_label",
        "suffix": None,
    },
    "is5t_eh": {"dir": "output_I-S5T_E+H", "col": "secondary_label", "suffix": None},
    "is5t_fov": {"dir": "output_I-S5T_FOV", "col": "secondary_label", "suffix": None},
}

ig_algmap = itemgetter("dir", "col", "suffix")


def _load_arff(fpath):
    # read arff file
    with open(fpath, "r") as fp:
        data = arff.load(fp)

    # convert to DataFrame
    _cols, _ = zip(*data["attributes"])
    etdata = pd.DataFrame(data["data"], columns=_cols)

    # convert to internat etdata format
    _rename = {"time": "t", "confidence": "status"}
    etdata.rename(columns=_rename, inplace=True)
    trackloss = np.all(etdata[["x", "y"]] == 0, axis=1)
    etdata["status"] = np.logical_and(etdata["status"] > 0.5, ~trackloss)
    etdata["t"] /= 1e6

    return etdata


def _sanity_check(etdata, gt, fpath=None):
    checks = (
        len(etdata) == len(gt),
        np.all(etdata[["t", "x", "y"]] == gt[["t", "x", "y"]]),
    )
    all_ok = all(checks)
    if not all_ok:
        print(f"Sanity check failed: {fpath}")

    return all_ok


def _map_events(evt):
    return TUM_EVENT_MAPPING.get(evt, 0)


##
# parsers
class TUMParser(object):
    dataset_param_mapping = {
        "GazeCom": {
            "dpath": "gazecom_annotations",
            "gtpath": "gaze_arff",
            "alg": alg_mapping_gazecom,
            "expert": ["handlabeller1", "handlabeller2", "handlabeller_final"],
        },
        "Hollywood2EM": {
            "dpath": "hollywood2_em",
            "gtpath": "ground_truth",
            "alg": alg_mapping_hollywood2_em,
            "expert": ["handlabeller_1", "handlabeller_final"],
        },
        "360EM": {
            "dpath": "360_em_dataset",
            "gtpath": "ground_truth",
            "alg": alg_mapping_360em,
            "expert": ["handlabeller_1_pl"],
        },
        "360EM-secondary": {
            "dpath": "360_em_dataset",
            "gtpath": "ground_truth",
            "alg": alg_mapping_360em_secondary,
            "expert": ["handlabeller_1_sl"],
        },
    }

    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        config = TUMParser.dataset_param_mapping.get(dataset_name)
        data_path = config.get("dpath")
        gt_path = config.get("gtpath")
        self.data_path = os.path.join(data_path, gt_path)
        self.gt_path = gt_path

        self.alg_params = utils.get_nested_item(
            TUMParser.dataset_param_mapping, [dataset_name, "alg"]
        )

        coders = utils.get_nested_item(
            TUMParser.dataset_param_mapping, [dataset_name, "expert"]
        )
        self.coder_params = {
            coder: {"dir": "ground_truth", "col": coder, "suffix": None}
            for coder in coders
        }

    def __call__(self, root, **kwargs):
        dataset_name = self.dataset_name
        data_dir = os.path.join(root, self.data_path)
        files = utils.dir_walk(data_dir, "arff")
        coder = kwargs.get("coder", "expert")
        params = self.alg_params if coder == "alg" else self.coder_params

        print(
            f"Parsing {dataset_name} ({coder}) from {data_dir}. This might take a while..."
        )
        data_accum = []
        for fpath in tqdm(files):
            fdir, fname = utils.split_path(fpath)
            etdata = _load_arff(fpath)

            for coder in params.keys():
                _dir, _col, _suffix = ig_algmap(params[coder])

                # load annotations
                _replace = {self.gt_path: _dir}
                if _suffix is not None:
                    _replace.update({fname: f"{fname}_{_suffix}"})
                fpath_alg = utils.multiple_replace(fpath, _replace)
                if not os.path.exists(fpath_alg):
                    print(f"MISSING: {fpath_alg}")
                    continue

                gt = _load_arff(fpath_alg)

                # sanity check
                if not _sanity_check(etdata, gt, fpath):
                    continue

                # parse algorithm labels
                evt = gt[_col].apply(_map_events)
                _etdata = copy.deepcopy(etdata[["t", "x", "y", "status"]])
                _etdata["evt"] = evt

                rdir = os.path.relpath(fdir, data_dir)
                spath = os.path.join(f"{dataset_name}_{coder}", rdir, fname)

                data_accum.append((_etdata, spath))

        return data_accum


def parse_gazecom(root, **kwargs):
    parser = TUMParser("GazeCom")
    result = parser(root, **kwargs)
    return result


def parse_hollywood2_em(root, **kwargs):
    parser = TUMParser("Hollywood2EM")
    result = parser(root, **kwargs)
    return result


def parse_360em(root, **kwargs):
    parser = TUMParser("360EM")
    result = parser(root, **kwargs)
    return result


def parse_360em_secondary(root, **kwargs):
    parser = TUMParser("360EM-secondary")
    result = parser(root, **kwargs)
    return result
