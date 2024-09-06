import argparse
import pickle
from datetime import datetime
import os

import sequential_learning.auxiliary as slaux
import sequential_learning.analysis as sla
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
    return parser


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    args.date = datetime.now()
    if args.sequence_ind >= len(slaux.shape_sequence) - 2:
        raise IOError(
            "ind {} is too high for list {}".format(
                args.sequence_ind, slaux.shape_sequence
            )
        )
    else:
        seq_start = args.sequence_ind
        seq_end = args.sequence_ind + 2
        shapes = slaux.shape_sequence[seq_start:seq_end]

    data_dict = slaux.load_shape_list(shapes)

    fig = slf.RelativeTransitionFigure(
        exper_data=data_dict,
        fig_folder=args.output_folder,
        region=args.region,
    )
    fig.panel_tasks()
    fig.panel_dec()
    fig.panel_subspace()
    fig.panel_learning()
    fig.panel_bhv_learning()

    shape_str = "-".join(shapes)
    fn = args.out_template.format(shape=shape_str, jobid=args.jobid)
    path = os.path.join(args.output_folder, fn)
    fig.save(path + ".pdf", use_bf="")
    analysis_results = fig.get_data()["main_analysis"]
    pickle.dump(analysis_results, open(path + ".pkl", "wb"))
