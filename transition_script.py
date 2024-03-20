import argparse
import pickle
from datetime import datetime
import matplotlib.pyplot as plt
import os

import sequential_learning.auxiliary as slaux
import sequential_learning.analysis as sla
import sequential_learning.visualization as slv


def create_parser():
    parser = argparse.ArgumentParser(
        description="perform decoding analyses on Kiani data"
    )
    parser.add_argument("--data_folder", default=slaux.BASEFOLDER)
    out_template = "transition_{shape}_{jobid}"
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
    parser.add_argument("--fwid", default=2, type=float)    
    return parser


shape_sequence = (
    "A2",
    "A3",
    "A4",
    "A3postA4",
    "A5",
    "A4postA5",
    "A3postA5",
    "A6",
    "A7",
    "A6postA7",
)


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    args.date = datetime.now()
    if args.sequence_ind >= len(shape_sequence) - 1:
        raise IOError(
            "ind {} is too high for list {}".format(args.sequence_ind, shape_sequence)
        )
    else:
        s1 = shape_sequence[args.sequence_ind]
        s2 = shape_sequence[args.sequence_ind + 1]

    data_dict = slaux.load_shape_list((s1, s2))

    shape_str = "{}-{}".format(s1, s2)
    out = sla.compute_cross_shape_generalization(
        data_dict[s1],
        data_dict[s2],
        args.winsize,
        args.tbeg,
        args.tend,
        args.winstep,
    )
    f, ax = plt.subplots(1, 1, figsize=(args.fwid, args.fwid))
    session_color = (0.7, 0.7, 0.7)
    shape_color = "r"

    slv.plot_cross_shape_generalization(
        *out,
        session_color=session_color,
        shape_color=shape_color,
        ax=ax,
    )

    fname = args.output_template.format(shape=shape_str, jobid=args.jobid)
    fpath = os.path.join(args.output_folder, fname)
    f.savefig(fpath + ".pdf", bbox_inches="tight", transparent=True)
    pickle.dump(out, open(fpath + ".pkl", "wb"))
