import argparse
from datetime import datetime

import sequential_learning.figures as slf
import sequential_learning.auxiliary as slaux


def create_parser():
    parser = argparse.ArgumentParser(
        description="perform decoding analyses on Kiani data"
    )
    parser.add_argument("--data_folder", default=slaux.BASEFOLDER)
    out_template = "prototype-extrap_{shape}_{region}_{add}_{jobid}"
    parser.add_argument(
        "-o",
        "--output_template",
        default=out_template,
        type=str,
        help="file to save the output in",
    )
    parser.add_argument(
        "--output_folder",
        default="../results/sequential_learning/extrap/",
        type=str,
        help="folder to save the output in",
    )
    parser.add_argument("--jobid", default="0000", type=str)
    parser.add_argument("--sequence_ind", default=0, type=int)
    parser.add_argument("--no_post", default=False, action="store_true")
    parser.add_argument("--uniform_resample", default=False, action="store_true")
    parser.add_argument("--region", default="IT", type=str)
    parser.add_argument("--use_early", default=False, action="store_true")
    parser.add_argument("--use_late", default=False, action="store_true")
    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()

    args.date = datetime.now()
    if args.no_post:
        use_seq = slaux.shape_sequence_nopost
    else:
        use_seq = slaux.shape_sequence
    if args.sequence_ind >= len(use_seq) - 1:
        raise IOError(
            "ind {} is too high for length {} list {}".format(
                args.sequence_ind, len(use_seq), use_seq
            )
        )
    else:
        shape = use_seq[args.sequence_ind]

    fig_class = slf.PrototypeBoundaryExtrapolationFigure
    add = []
    if args.use_early:
        fig_class = slf.EarlyPrototypeBoundaryExtrapolationFigure
        add.append("early")
    elif args.use_late:
        fig_class = slf.LatePrototypeBoundaryExtrapolationFigure
        add.append("late")
    fig = fig_class(
        shape=shape,
        region=args.region,
    )
    fig.panel_pattern()

    fname = args.output_template.format(
        shape=shape,
        jobid=args.jobid,
        region=args.region,
        add="-".join(add),
    )
    fig.save(fname + ".pdf", use_bf=args.output_folder)
    
