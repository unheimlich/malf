import os
from argparse import ArgumentParser
import nibabel as nib
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt
from scipy.ndimage.filters import gaussian_filter, convolve
__author__ = 'ajesson'


def main():

    parser = get_parser()
    args = parser.parse_args()

    tmp_dir = args.out_dir + 'tmp/'

    if not os.path.exists(tmp_dir):

        os.makedirs(tmp_dir)

    register_templates(args.t1_file, args.atlas_dir, tmp_dir)

    id_list = next(os.walk(args.atlas_dir))[1]

    x, x_templates, y_templates, affine = load_files(args.t1_file, tmp_dir, id_list)

    p_templates = log_odds(y_templates)

    if args.method == 'majority':

        y_pred = np.argmax(p_templates.sum(0), 0)

    else:
        d = (x_templates - x)**2
        sigma = d.mean(dtype=np.float64)
        l = np.exp(-0.5*d/sigma, dtype=np.float32)
        p = p_templates*l
        y_pred = np.argmax(p.sum(0), 0)

        if args.method == 'semi_local':

            y_pred = semi_local(l, p_templates, y_pred, args.beta)

    out_img = nib.Nifti1Image(np.uint8(y_pred), affine)
    # out_img = nib.Nifti1Image(np.uint8(m), affine)

    nib.save(out_img, args.out_dir + 'malf_labels.nii.gz')


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
                        choices=['majority', 'local', 'global', 'semi_local'])

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


def register_templates(t1_file, atlas_dir, out_dir):

    id_list = next(os.walk(atlas_dir))[1]

    for template_id in id_list:

        affine_xfm = out_dir + template_id + '_0GenericAffine.mat'
        warp = out_dir + template_id + '_1Warp.nii.gz'
        warped = out_dir + template_id + '_Warped.nii.gz'
        i_warp = out_dir + template_id + '_1InverseWarp.nii.gz'
        i_warped = out_dir + template_id + '_InverseWarped.nii.gz'
        labels = out_dir + template_id + '_labels.nii.gz'

        if not(os.path.isfile(warped) and os.path.isfile(labels)):
            os.system('bash ants/antsRegistrationSyNQuick.sh -d 3 -n 8 -f ' + t1_file +
                      ' -m ' + atlas_dir + template_id + '/T1_norm-stx152lin.nii.gz -o ' +
                      out_dir + template_id + '_')
            os.system('ants/WarpImageMultiTransform 3 ' + atlas_dir + template_id + '/labels.nii.gz ' +
                      labels + ' -R ' + t1_file + ' --use-NN ' +
                      warp + ' ' + affine_xfm)
            os.system('rm ' + affine_xfm + ' ' + warp + ' ' + i_warp + ' ' + i_warped)


def load_files(t1_file, tmp_dir, id_list):

    t1_img = nib.load(t1_file)
    affine = t1_img.get_affine()
    x = np.zeros((1, 1) + t1_img.shape, dtype=np.float32)
    x[0, 0] = t1_img.get_data()

    x_template = np.zeros((len(id_list), 1) + t1_img.shape, dtype=np.float32)
    y_template = np.zeros((len(id_list), 1) + t1_img.shape, dtype=np.uint8)

    for i, template_id in enumerate(id_list):

        template_img = nib.load(tmp_dir + template_id + '_Warped.nii.gz')
        x_template[i, 0] = template_img.get_data()
        label_img = nib.load(tmp_dir + template_id + '_labels.nii.gz')
        y_template[i, 0] = label_img.get_data()

    return x, x_template, y_template, affine


def log_odds(y):

    num_labels = len(np.unique(y))
    num_templates = y.shape[0]
    p = np.zeros((num_templates, num_labels) + y.shape[2:], dtype=np.float16)

    for i in range(num_templates):

        for j in range(num_labels):

            mask = y[i, 0] == j
            # mask_int = y[i, 0] == j
            # mask_ext = y[i, 0] != j
            #
            # d_int = np.exp(distance_transform_edt(mask_int))
            # d_ext = np.exp(-distance_transform_edt(mask_ext))
            #
            # p_tmp[0, j, mask_int] = d_int[mask_int]
            # p_tmp[0, j, mask_ext] = d_ext[mask_ext]
            p[i, j] = gaussian_filter(np.float32(mask), sigma=0.667) + np.finfo(np.float16).eps
            # p[i, j] = np.float32(mask)
            print i

    return p


def semi_local(l, p_templates, L, beta):

    q = l/(l.sum(0) + np.finfo(np.float32).eps)

    k = np.ones((1, 1, 3, 3, 3), dtype=np.float32)

    diff = np.inf

    m_last = np.zeros(L.shape, dtype=np.uint8)

    while diff > 1000000:

        for i in range(l.shape[0]):

            p_template = np.zeros((1, 1) + L.shape)

            for j in range(p_templates.shape[1]):

                mask = L == j

                p_template[0, 0, mask] += p_templates[i, j, mask]

            q[i] = l[i:i+1]*p_template*np.exp(beta*convolve(q[i:i+1], k, mode='constant'))

        q = q/(q.sum(0) + np.finfo(np.float32).eps)
        m = np.argmax(q, 0)[0]
        diff = (m != m_last).sum()
        m_last = m

        print diff

    p = p_templates*q
    L = np.argmax(p.sum(0), 0)

    return L

if __name__ == "__main__":
    main()
