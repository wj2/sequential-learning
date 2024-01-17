
import argparse
import pickle
from datetime import datetime
import os

import sequential_learning.figures as slf
import sequential_learning.auxiliary as slaux


def create_parser():
    parser = argparse.ArgumentParser(
        description='perform decoding analyses on Kiani data'
    )
    parser.add_argument("--data_folder", default=slaux.BASEFOLDER)
    out_template = "summary_{shape}_{jobid}"
    parser.add_argument('-o', '--output_template', default=out_template, type=str,
                        help='file to save the output in')
    parser.add_argument(
        '--output_folder',
        default="../results/sequential_learning/summary_figs/",
        type=str,
        help='folder to save the output in'
    )
    parser.add_argument("--jobid", default="0000", type=str)
    parser.add_argument("--shape_ind", default=0, type=int)
    return parser


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()

    args.date = datetime.now()
    shapes = slaux.get_shape_folders(args.data_folder)
    if args.shape_ind >= len(shapes):
        raise IOError("ind {} is too high for list {}".format(
            args.shape_ind, shapes
        ))
    else:
        shape = shapes[args.shape_ind]
    print(shapes)
    print(shape)

    ss_fig = slf.ShapeSpaceSummary(shape)
    ss_fig.panel_bhv()
    
    ss_fig.panel_decoding()

    fname = args.out_template.format(shape=shape, jobid=args.jobid)
    ss_fig.save(fname + ".pdf", use_bf=args.output_folder)
    ss_fig.save(fname + ".svg", use_bf=args.output_folder)
