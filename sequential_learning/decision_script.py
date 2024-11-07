import argparse
import pickle
from datetime import datetime
import os

import sequential_learning.auxiliary as slaux
import sequential_learning.figures as slf


def create_parser():
    parser = argparse.ArgumentParser(
        description="perform decoding analyses on Kiani data"
    )
    parser.add_argument("--data_folder", default=slaux.BASEFOLDER)
    out_template = "decision_{shape}_{jobid}"
    parser.add_argument(
        "-o",
        "--output_template",
        default=out_template,
        type=str,
        help="file to save the output in",
    )
    parser.add_argument(
        "--output_folder",
        default="../results/sequential_learning/decision/",
        type=str,
        help="folder to save the output in",
    )
    parser.add_argument("--winsize", default=500, type=float)
    parser.add_argument("--winstep", default=500, type=float)
    parser.add_argument("--tbeg", default=0, type=float)
    parser.add_argument("--tend", default=500, type=float)
    parser.add_argument("--jobid", default="0000", type=str)
    parser.add_argument("--sequence_ind", default=0, type=int)
    parser.add_argument("--previous_shapes", default=1, type=int)
    parser.add_argument("--region", default="IT")
    parser.add_argument("--use_post", default=False, action="store_true")
    parser.add_argument("--uniform_resample", default=False, action="store_true")
    parser.add_argument("--no_video", default=False, action="store_true")
    parser.add_argument("--min_trials", default=100, type=int)
    parser.add_argument(
        "--exclude_choice_dimension", default=False, action="store_true"
    )
    parser.add_argument("--only_choice_dimension", default=False, action="store_true")
    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()

    args.date = datetime.now()
    if not args.use_post:
        use_seq = slaux.shape_sequence_nopost
    else:
        use_seq = slaux.shape_sequence
    if args.sequence_ind >= len(use_seq) - 1 - args.previous_shapes:
        raise IOError(
            "ind {} is too high for length {} list {}".format(
                args.sequence_ind, len(use_seq), use_seq
            )
        )
    else:
        seq_start = args.sequence_ind
        seq_end = args.sequence_ind + args.previous_shapes + 1
        shapes = use_seq[seq_start:seq_end]
    use_fields = ("cat_proj", "anticat_proj")
    if not args.exclude_choice_dimension:
        use_fields = ("chosen_cat",) + use_fields
    if args.only_choice_dimension:
        use_fields = ("chosen_cat",)

    fig = slf.DecisionLearningFigure(
        shapes=shapes,
        fig_folder=args.output_folder,
        region=args.region,
        uniform_resample=args.uniform_resample,
        min_trials=args.min_trials,
        use_fields=use_fields,
    )
    fig.panel_proj()
    fig.panel_overlap()

    shape_str = "train{}_gen{}".format(fig.pre_key, fig.gen_key)
    fn = args.output_template.format(shape=shape_str, jobid=args.jobid)
    path = os.path.join(args.output_folder, fn)
    fig.save(path + ".pdf", use_bf="")
    analysis_results = fig.get_data().get("main_analysis")
    pickle.dump(analysis_results, open(path + ".pkl", "wb"))
