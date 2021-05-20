import os
import argparse
import json
from distutils.dir_util import mkpath

from tqdm import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from misc import utils
from misc import eval_utils
from misc import matching

plt.ion()
plt.rc("axes.spines", top=False, right=False)


def get_arguments():
    parser = argparse.ArgumentParser(description="Eye-movement event kpi test")
    parser.add_argument(
        "-job", type=str, default=None, required=True, help="JSON job definition file"
    )
    parser.add_argument(
        "--output", type=str, default="results", help="Output directory"
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        default=False,
        help="Plot mode. Overwritten if specified in job file",
    )
    return parser.parse_args()


##
# run evaluation
if __name__ == "__main__":
    args = get_arguments()
    root_repo = os.path.join(os.path.dirname(__file__))
    odir = utils.path2abs(args.output, root_repo)
    mkpath(odir)

    # load jobs
    jpath = utils.path2abs(args.job, root_repo)
    with open(jpath, "r") as f:
        jobs = json.load(f)
    # with open(jpath, 'w') as f:
    #     json.dump(jobs, f, indent=4)

    # get config
    _root, _ = utils.split_path(jpath)
    root = jobs.get("root", _root)
    event_map_default = jobs.get("event_map", {})
    multiclass_strategy_default = jobs.get("multiclass_strategy", ["all"])
    binary_strategy_default = jobs.get("binary_strategy", [])
    plot_mode = jobs.get("plot_mode", args.plot)

    # run jobs
    result_accum = []
    for job in jobs.get("jobs", []):
        files = utils.get_file_list(job, root)
        matchers = job.get("matchers", {})
        event_map, event_labels = utils.get_event_map(job, event_map_default)
        multiclass_strategy = job.get(
            "multiclass_strategy", multiclass_strategy_default
        )
        binary_strategy = job.get("binary_strategy", binary_strategy_default)
        job_label = job.get("label", None)

        meta_accum = []
        iterator = tqdm(list(zip(*files)), desc=job.get("pr"))
        for fpath_gt, fpath_pr in iterator:
            # init
            fname = os.path.relpath(fpath_gt, os.path.join(root))
            _gt, _pr = job.get("gt"), job.get("pr")
            meta = {"gt": _gt, "pr": _pr, "fname": fname}

            # unittests
            unittests = {}
            if job.get("unittest", False):
                _, _fname = utils.split_path(fname)
                upath = os.path.join(root, _gt, "unittests", _pr, f"{_fname}.json")
                if os.path.exists(upath):
                    with open(upath, "r") as f:
                        unittests = json.load(f)
                        print(f"Running unittest on {fname}")
                else:
                    print(f"No unittest for {fname}")

            # load data
            data_gt, data_pr = utils.load_data(fpath_gt, fpath_pr, event_map)
            event_matcher = matching.EventMatcher(gt=data_gt, pr=data_pr)

            # run eval
            for matcher, matching_kwargs in matchers.items():
                matcher_label = filter(None, (matcher, job_label))
                matcher_label = {"matcher": "-".join(matcher_label)}

                if plot_mode:
                    match_plot_kwargs = job.get("match-plot-kwargs", None)
                    kwargs = (
                        matching_kwargs
                        if isinstance(matching_kwargs, list)
                        else [matching_kwargs]
                    )
                    # add plot mode indicator
                    _plot_mode = {"plot-mode": True}
                    kwargs = [
                        utils.merge_dicts([_kwargs, _plot_mode]) for _kwargs in kwargs
                    ]

                    # run matching
                    _match_result = [
                        event_matcher.run_matching(matcher, **_kwargs)
                        for _kwargs in kwargs
                    ]
                    _, events = zip(*_match_result)

                    # interactive plot
                    utils.plot_job(
                        matcher=event_matcher,
                        events=events,
                        spath=os.path.relpath(fpath_gt, root),
                        odir=odir,
                        match_plot_kwargs=match_plot_kwargs,
                        matcher_label=matcher_label["matcher"],
                        data=data_gt,
                    )
                    continue

                unittest = unittests.get(matcher_label["matcher"])
                eval_result = eval_utils.calc_scores(
                    event_matcher=event_matcher,
                    matcher=matcher,
                    matching_kwargs=matching_kwargs,
                    labels=event_labels,
                    multiclass_strategy=multiclass_strategy,
                    binary_strategy=binary_strategy,
                    meta=[meta, matcher_label],
                    unittest=unittest,
                )

                result_accum.extend(eval_result)

    # save result
    if not plot_mode:
        result = pd.DataFrame(result_accum)
        _, jname = utils.split_path(jpath)
        rpath = os.path.join(odir, f"{jname}.csv")
        result.to_csv(rpath, index=False)

        # aggregate result
        result_agg = result.groupby(
            ["gt", "pr", "matcher", "eval", "event"], as_index=False
        ).agg(np.nanmean)
        result_agg.to_csv(os.path.join(odir, f"{jname}-agg.csv"), index=False)
