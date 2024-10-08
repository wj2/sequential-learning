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
    parser.add_argument("--sequence_length", default=2, type=int)
    parser.add_argument("--region", default="IT")
    parser.add_argument("--use_screen_feature", default=False, action="store_true")
    
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
        seq_start = args.sequence_ind
        seq_end = args.sequence_ind + args.sequence_length
        shapes = slaux.shape_sequence[seq_start : seq_end]

    data_dict = slaux.load_shape_list(shapes)

    data_list = list(data_dict.values())
    if args.use_binary_feature is not None:
        if args.use_screen_feature:
            feat_field = "stim_feature_screen"
            feat_str = "screenfeat"
        else:
            feat_field = "stim_feature_MAIN"
            feat_str = "feat"
        masks = slaux.get_binary_feature_masks(
            *data_list, feat_ind=args.use_binary_feature, feat_field=feat_field,
        )
        boundary = "{}-{}".format(feat_str, args.use_binary_feature)
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
        region=args.region,
    )

    shape_str = "-".join(shapes)
    fn = args.output_template.format(
        shape=shape_str, jobid=args.jobid, boundary=boundary
    )
    path = os.path.join(args.output_folder, fn)
    info = out[:3]
    gen = out[3]
    var = out[4]
    out_dict = {"args": vars(args), "info": info, "gen": gen, "var": var}
    pickle.dump(out_dict, open(path + ".pkl", "wb"))

    f = slf.SpecificTransitionFigure(info, gen, var)
    f.panel_var()
    f.panel_gen()    
    f.save(path + ".pdf", use_bf="")
