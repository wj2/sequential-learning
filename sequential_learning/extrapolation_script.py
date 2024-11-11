import argparse
from datetime import datetime

import sequential_learning.figures as slf
import sequential_learning.auxiliary as slaux


def create_parser():
    parser = argparse.ArgumentParser(
        description="perform decoding analyses on Kiani data"
    )
    parser.add_argument("--data_folder", default=slaux.BASEFOLDER)
    out_template = "extrap_{dec_field}x{gen_field}_{shape}_{jobid}"
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
    parser.add_argument("--dec_field", default="chosen_cat", type=str)
    parser.add_argument("--gen_field", default="anticat_proj", type=str)
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
    if args.dec_field == "cat_proj":
        dec_ref = 0
    elif args.dec_field == "chosen_cat":
        dec_ref = 1.5

    fig = slf.BoundaryExtrapolationFigure(
        shape=shape,
        dec_field=args.dec_field,
        gen_field=args.gen_field,
        dec_ref=dec_ref,
    )
    fig.panel_pattern()

    fname = args.output_template.format(
        shape=shape,
        dec_field=args.dec_field,
        gen_field=args.gen_field,
        jobid=args.jobid,
    )
    fig.save(fname + ".pdf", use_bf=args.output_folder)
