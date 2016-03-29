import os
from argparse import ArgumentParser
__author__ = 'ajesson'


def main():

    parser = get_parser()
    args = parser.parse_args()


def get_parser():

    usage = "%(prog)s [options] -t1 <patient_t1_file> -o <output_directory>" \
            " -a <atlas_directory>"
    parser = ArgumentParser(prog='malf', usage=usage)

    parser.add_argument("-t1", "--t1_file", dest="t1_file",
                        help="Patient T1 file", required=True)

    parser.add_argument("-a", "--atlas_directory", dest="atlas_dir",
                        help="Directory of template and label images",
                        required=True)

    parser.add_argument("-o", "--output_directory", dest="out_dir",
                        help="Directory to write output files",
                        required=True)

    parser.add_argument("-m", "--method", dest="method",
                        help="label fusion method",
                        required=False, default='majority',
                        choices=['majority', 'local', 'global', 'semi_global'])

    parser.add_argument("-l", "--likelihood", dest="likelihood",
                        help="Template patient similarity metric",
                        required=False, default='gaussian',
                        choices=['gaussian', 'ncc', 'nmi'])

    parser.add_argument("-beta", "--beta", dest="beta",
                        help="MRF beta parameter",
                        required=False, default=0.75)

    parser.add_argument("-mask", "--mask_file", dest="mask_file",
                        help="Patient mask file", required=False)

    return parser

if __name__ == "__main__":
    main()
