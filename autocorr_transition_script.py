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
    out_template = "transition_full_{boundary}_{shape}_{jobid}"
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
    parser.add_argument("--fwid", default=6, type=float)
    parser.add_argument("--no_indiv_zscore", default=False, action="store_true")
    parser.add_argument("--use_binary_feature", default=None, type=int)
    parser.add_argument("--use_pre_post_data", default=False, action="store_true")    
    return parser


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    args.date = datetime.now()
    if args.sequence_ind >= len(slaux.shape_sequence) - 1:
        raise IOError(
            "ind {} is too high for list {}".format(
                args.sequence_ind, slaux.shape_sequence
            )
        )
    elif args.use_pre_post_data:
        shapes = list(slaux.sequence_groups.values())[args.sequence_ind]
    else:
        shapes = slaux.shape_sequence[args.sequence_ind:args.sequence_ind + 2]

    data_dict = slaux.load_shape_list(shapes)

    data_list = list(data_dict.values())
    if args.use_binary_feature is not None:
        masks = slaux.get_binary_feature_masks(
            *data_list, feat_ind=args.use_binary_feature
        )
        boundary = "feat-{}".format(args.use_binary_feature)
    else:
        masks = slaux.get_prototype_masks(*data_list, data_ind=1)
        boundary = "proto-{}".format(shapes[1])
    data_mask_pairs = list(zip(data_list, masks))
    out = sla.cross_session_generalization(
        *data_mask_pairs,
        stepsize=args.winstep,
        winsize=args.winsize,
        tbeg=args.tbeg,
        tend=args.tend,
        indiv_zscore=not args.no_indiv_zscore,
    )

    shape_str = "-".join(shapes)
    fn = args.output_template.format(
        shape=shape_str, jobid=args.jobid, boundary=boundary
    )
    path = os.path.join(args.output_folder, fn)
    out_dict = {"args": vars(args), "gen": out}
    pickle.dump(out_dict, open(path + ".pkl", "wb"))

    f, ax = plt.subplots(1, 1, figsize=(args.fwid, args.fwid))
    slv.plot_decoder_autocorrelation(*out, ax=ax)
    f.savefig(path + ".pdf", bbox_inches="tight", transparent=True)
