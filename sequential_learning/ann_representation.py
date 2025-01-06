import numpy as np
import sklearn.decomposition as skd
import sklearn.preprocessing as skp
import sklearn.svm as skm

import sequential_learning.auxiliary as slaux
import general.torch.pretrained as gtp
import general.plotting as gpl
import general.neural_analysis as na


def get_representations(shapes=slaux.shape_sequence_unique, **kwargs):
    imgs_all = slaux.load_all_images(shapes)
    net = gtp.GenericPretrainedNetwork()

    out_reps = {}
    for shape, (lv1, lv2, imgs) in imgs_all.items():
        reps = net.get_representation(imgs)
        lvs = np.stack((lv1, lv2), axis=1)
        reps = np.reshape(reps.detach().numpy(), (reps.shape[0], -1))
        out_reps[shape] = (lvs, imgs, reps)
    return out_reps


def project_lvs(ann_reps, data_dict, div_feats=1000):
    reps_new = {}
    for shape, (lvs, imgs, reps) in ann_reps.items():
        data_shape = data_dict.get(shape)
        if data_shape is not None:
            lvs_norm = lvs / div_feats
            angle = slaux.get_common_angle(data_shape["cat_def_MAIN"][0].to_numpy())
            new_lvs = np.stack(slaux.project_on_angle(lvs_norm, angle), axis=1)
            reps_new[shape] = (new_lvs, imgs, reps)
        else:
            print("no data for shape {}".format(shape))
    return reps_new


@gpl.ax_adder(three_dim=True)
def visualize_shape_overlap(*img_reps, cmaps=None, ax=None, lv_ind=0):
    if cmaps is None:
        cmaps = ("magma", "viridis") * len(img_reps)
    full = []
    for _, _, reps_i in img_reps:
        full.append(skp.StandardScaler().fit_transform(reps_i))
    trs = skd.PCA(3).fit(np.concatenate(full, axis=0))
    for i, (lv_i, _, rep_i) in enumerate(img_reps):
        pts = trs.transform(rep_i)
        ax.scatter(*pts.T, c=lv_i[:, lv_ind], cmap=cmaps[i])
    ax.set_aspect("equal")


def generalization_pattern(
    lvs,
    reps,
    dec_ind=0,
    gen_ind=1,
    n_folds=20,
    model=skm.LinearSVC,
    test_frac=.1,
    **kwargs,
):

    reps = np.expand_dims(reps.T, -1)
    labels = lvs[:, dec_ind] > 0
    mask1 = lvs[:, gen_ind] > 0
    mask2 = np.logical_not(mask1)
    
    pipe = na.make_model_pipeline(model=model, **kwargs)
    dec1 = na.nominal_fold(
        reps[:, mask1],
        labels[mask1],
        pipe,
        n_folds,
        return_projection=True,
        c_gen=reps[:, mask2],
        l_gen=labels[mask2],
        test_frac=test_frac,
    )
    dec2 = na.nominal_fold(
        reps[:, mask2],
        labels[mask2],
        pipe,
        n_folds,
        return_projection=True,
        c_gen=reps[:, mask1],
        l_gen=labels[mask1],
        test_frac=test_frac,
    )
    gen1_proj = np.mean(dec1["projection_gen"], axis=(0, -1))
    lv1_proj = lvs[mask2]
    gen2_proj = np.mean(dec2["projection_gen"], axis=(0, -1))
    lv2_proj = lvs[mask1]
    gen_proj = np.concatenate((gen1_proj, gen2_proj), axis=0)
    lv_proj = np.concatenate((lv1_proj, lv2_proj), axis=0)
    return lv_proj, gen_proj


