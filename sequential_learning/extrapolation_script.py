import argparse
from datetime import datetime

import sequential_learning.figures as slf
import sequential_learning.auxiliary as slaux


def create_parser():
    parser = argparse.ArgumentParser(
        description="perform decoding analyses on Kiani data"
    )
    parser.add_argument("--data_folder", default=slaux.BASEFOLDER)
    out_template = "{kind}_{dec_field}-x-{gen_field}_{shape}_{jobid}"
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
    parser.add_argument("--dec_field", default="cat_proj", type=str)
    parser.add_argument("--gen_field", default="anticat_proj", type=str)
    parser.add_argument("--use_prototypes", default=False, action="store_true")
    parser.add_argument("--balance_complement", default=False, action="store_true")
    parser.add_argument("--no_fixation", default=False, action="store_true")
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
        balance_field = "chosen_cat"
        dec_ref = 0
    elif args.dec_field == "chosen_cat":
        balance_field = "cat_proj"
        dec_ref = 1.5

    if args.balance_complement:
        use_balance_field = balance_field
    else:
        use_balance_field = None

    if args.use_prototypes:
        gen_func = slaux.proto_box_mask
        gen_str = "prototype"
    else:
        gen_func = None
        gen_str = args.gen_field

    
    if args.no_fixation:
        fig_class = slf.BoundaryExtrapolationFigure
    else:
        fig_class = slf.FixationBoundaryExtrapolationFigure
        
    fig = fig_class(
        shape=shape,
        dec_field=args.dec_field,
        gen_field=args.gen_field,
        gen_func=gen_func,
        balance_field=use_balance_field,
        dec_ref=dec_ref,
    )
    fig.panel_pattern()

    fname = args.output_template.format(
        shape=shape,
        dec_field=args.dec_field,
        gen_field=gen_str,
        jobid=args.jobid,
        kind="extrap",
    )
    fig.save(fname + ".pdf", use_bf=args.output_folder)
    exper_data = fig.get_data()["exper_data"]

    slf.DecoderErrorPatternFigure(
        shape=shape,
        dec_field=args.dec_field,
        balance_field=use_balance_field,
        dec_ref=dec_ref,
        exper_data=exper_data[0],
    )
    fig.panel_pattern()

    fname = args.output_template.format(
        shape=shape,
        dec_field=args.dec_field,
        gen_field=gen_str,
        jobid=args.jobid,
        kind="full",
    )
    fig.save(fname + ".pdf", use_bf=args.output_folder)

    try:
        fig = slf.ANNBoundaryExtrapolationFigure(
            shape=shape,
            dec_field=args.dec_field,
            gen_field=args.gen_field,
            gen_func=gen_func,
            balance_field=use_balance_field,
            dec_ref=dec_ref,
            exper_data=exper_data[0]
        )
        fig.panel_pattern()

        fname = args.output_template.format(
            shape=shape,
            dec_field=args.dec_field,
            gen_field=gen_str,
            jobid=args.jobid,
            kind="ann",
        )
        fig.save(fname + ".pdf", use_bf=args.output_folder)
    except Exception as e:
        print(e)
        print("ANN analysis failed")
