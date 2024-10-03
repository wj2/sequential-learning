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
    out_template = "rep_full_{shape}_{jobid}"
    parser.add_argument(
        "-o",
        "--output_template",
        default=out_template,
        type=str,
        help="file to save the output in",
    )
    parser.add_argument(
        "--output_folder",
        default="../results/sequential_learning/transition/",
        type=str,
        help="folder to save the output in",
    )
    parser.add_argument("--winsize", default=500, type=float)
    parser.add_argument("--winstep", default=500, type=float)
    parser.add_argument("--tbeg", default=0, type=float)
    parser.add_argument("--tend", default=500, type=float)
    parser.add_argument("--jobid", default="0000", type=str)
    parser.add_argument("--sequence_ind", default=0, type=int)
    parser.add_argument("--region", default="IT")
    parser.add_argument("--no_post", default=False, action="store_true")
    parser.add_argument("--uniform_resample", default=False, action="store_true")
    parser.add_argument("--no_video", default=False, action="store_true")
    parser.add_argument("--min_trials", default=100, type=int)
    return parser


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    args.date = datetime.now()
    if args.no_post:
        use_seq = slaux.shape_sequence_nopost
    else:
        use_seq = slaux.shape_sequence
    if args.sequence_ind >= len(use_seq) - 2:
        raise IOError(
            "ind {} is too high for length {} list {}".format(
                args.sequence_ind, len(use_seq), use_seq
            )
        )
    else:
        seq_start = args.sequence_ind
        seq_end = args.sequence_ind + 2
        shapes = use_seq[seq_start:seq_end]

    fig = slf.RelativeTransitionFigure(
        shapes=shapes,
        fig_folder=args.output_folder,
        region=args.region,
        uniform_resample=args.uniform_resample,
        save_video=not args.no_video,
        min_trials=args.min_trials,
    )
    fig.panel_tasks()
    fig.panel_dec()
    fig.panel_subspace()
    fig.panel_learning()
    fig.panel_bhv_learning()
    fig.panel_proj_learning()

    shape_str = "-".join(shapes)
    fn = args.output_template.format(shape=shape_str, jobid=args.jobid)
    path = os.path.join(args.output_folder, fn)
    fig.save(path + ".pdf", use_bf="")
    analysis_results = fig.get_data().get("main_analysis")
    pickle.dump(analysis_results, open(path + ".pkl", "wb"))
