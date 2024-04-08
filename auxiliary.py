import os
import scipy.io as sio
import skimage.io as skio
import re
import numpy as np
import pandas as pd
import scipy.stats as sts
import sklearn.model_selection as skms

import general.utility as u
import general.data_io as gio


BASEFOLDER = "../data/sequential_learning/data/"


ft1 = (
    "A(?P<date>[0-9a-z]+)\\.(?P<region>[A-Z0-9+]+)\\.learn(?P<shape>A[0-9])"
    "\\-d(?P<day>[0-9]+)\\.[0-9]+\\.FIRA\\.mat"
)
ft2 = (
    "A(?P<date>[0-9a-z]+)\\.(?P<shape>A[0-9])\\-d(?P<day>[0-9]+)(\\.postA[0-9]+)?"
    "\\.FIRA\\.LMAN_"
    "cat\\.(?P<region>[A-Z0-9+]+)\\.mat"
)
file_templates = (ft1, ft2)

default_type_dict = {
    "time": float,
    "value": float,
}


stim_temp = "(?P<lv1>-?[0-9]+)_(?P<lv2>-?[0-9]+)_stim\\.png"


sequence_groups = {
    "A3-A4": ("A3", "A4", "A3postA4"),
    "A4-A5": ("A4", "A5", "A4postA5"),
    "A5-A6": ("A5", "A6", "A5postA6"),
    "A6-A7": ("A6", "A7", "A6postA7"),
}


def load_shape_list(
    shapes,
    data_folder=BASEFOLDER,
    sort_by="day",
    max_files=np.inf,
    exclude_invalid=True,
    **kwargs,
):
    out_data = {}
    for shape in shapes:
        data = gio.Dataset.from_readfunc(
            load_kiani_data_folder,
            os.path.join(data_folder, shape),
            max_files=max_files,
            sort_by=sort_by,
        )
        if exclude_invalid:
            data = filter_valid(data)
        out_data[shape] = data
    return out_data


def sample_uniform_mask(
    data,
    cat_proj="cat_proj",
    anticat_proj="anticat_proj",
    proj_range=None,
    n_bins=5,
    eps=1e-10,
):
    cats = data[cat_proj]
    anti_cats = data[anticat_proj]
    if proj_range is None:
        p_end = np.max(list(np.max(np.abs(x)) for x in cats))
        p_end = np.min([.71, p_end])
        proj_range = (-p_end - eps, p_end + eps)
    edges = (np.linspace(*proj_range, n_bins + 1),) * 2
    masks = []
    for i, cat_i in enumerate(cats):
        anti_cat_i = anti_cats[i]
        sample_pos = np.stack((cat_i, anti_cat_i), axis=1)

        counts, edges, binnumber = sts.binned_statistic_dd(
            sample_pos,
            np.ones(len(sample_pos)),
            statistic="count",
            bins=edges,
        )
        _, counts = np.unique(binnumber, return_counts=True)
        trl_samps = np.min(counts)
        
        if trl_samps > 1:
            splitter = skms.StratifiedShuffleSplit(
                n_splits=1,
                train_size=int(trl_samps) * n_bins**2,
            )
            samp_inds, _ = list(splitter.split(sample_pos, binnumber))[0]
        else:
            samp_inds = []
        mask_i = np.isin(np.arange(len(cat_i)), samp_inds)
        masks.append(mask_i)
    return masks


def get_shape_folders(use_folder=BASEFOLDER, pattern="A[0-9]+[a-z]?"):
    fls = os.listdir(use_folder)
    out = list(fl for fl in fls if re.match(pattern, fl) is not None)
    return out


def load_stim_images(stim_folder, template=stim_temp):
    fls = os.listdir(stim_folder)
    lv1s = []
    lv2s = []
    imgs = []
    for fl in fls:
        m = re.match(template, fl)
        if m is not None:
            lv1 = int(m.group("lv1"))
            lv2 = int(m.group("lv2"))
            fp = os.path.join(stim_folder, fl)
            img = skio.imread(fp)
            lv1s.append(lv1)
            lv2s.append(lv2)
            imgs.append(img)
    return np.array(lv1s), np.array(lv2s), np.stack(imgs, axis=0)


def load_file(fname, type_dict=default_type_dict):
    data = sio.loadmat(fname)
    fira = data["FIRA"]
    keys, types = fira[0, 0]["events"][0, 0]
    vals = fira[0, 1]
    full_dict = {}
    for i, k in enumerate(keys):
        if u.check_list(k):
            k = k[0]
        vals_i = vals[:, i]
        use_type = type_dict.get(types[i][0], object)
        if np.product(vals_i[0].shape) != 1 or k == "fp_wd_type":
            use_type = object

        vals_use = np.zeros(len(vals_i), dtype=use_type)
        for j, vj in enumerate(vals_i):
            vuj = np.squeeze(vj)[()]
            if u.check_list(vuj) and len(vuj) == 0:
                vals_use[j] = np.nan
            else:
                try:
                    vals_use[j] = vuj
                except ValueError:
                    print(k, use_type, vuj)
                    vals_use[j] = np.nan
        full_dict[k] = vals_use
    spike_times_all = fira[0, 2][:, 0]
    spks = np.zeros(
        (spike_times_all.shape[0], spike_times_all[0].shape[0]), dtype=object
    )
    for i, spk_trl in enumerate(spike_times_all):
        spks[i] = list(np.squeeze(neur_trl) for neur_trl in spk_trl[:, 0])
    full_dict["spikeTimes"] = list(s for s in spks)
    return full_dict


def _get_projs(
    data,
    stim_feat_field="stim_feature_MAIN",
    cat_bound_field="cat_def_MAIN",
):
    feats = np.zeros((len(data), 2))
    feats[:] = np.nan
    for i, f_i in enumerate(data[stim_feat_field].to_numpy()):
        if u.check_list(f_i) and len(f_i) == 2:
            feats[i] = f_i
    feats = feats / 1000
    cat_boundary_angles, counts = np.unique(
        data[cat_bound_field].to_numpy(), return_counts=True
    )
    cba = cat_boundary_angles[np.argmax(counts)]

    cat_vec = u.make_unit_vector(
        np.array(
            [
                np.cos(np.radians(cba)),
                -np.sin(np.radians(cba)),
            ]
        )
    )
    anti_cat_vec = u.make_unit_vector(
        np.array(
            [
                np.cos(np.radians(cba)),
                np.sin(np.radians(cba)),
            ]
        )
    )
    cat = feats @ cat_vec
    anticat = feats @ anti_cat_vec
    return cat, anticat


def load_kiani_data_folder(
    folder, templates=file_templates, monkey_name="Z", max_files=np.inf
):
    datas = []
    n_neurs = []
    dates = []
    monkeys = []
    days = []
    shapes = []

    files_loaded = 0
    file_gen = u.load_folder_regex_generator(folder, *templates, load_func=load_file)
    for fl, fl_info, data_fl in file_gen:
        dates.append(fl_info["date"])
        shapes.append(fl_info["shape"])
        days.append(int(fl_info["day"]))
        monkeys.append(monkey_name)

        n_neur_fl = data_fl["spikeTimes"][0].shape[0]
        if fl_info["region"] == "V4+FEF":
            fef_labels = ("FEF",) * 32
            v4_labels = ("V4",) * (n_neur_fl - 32)
            labels = fef_labels + v4_labels
            data_fl["neur_regions"] = (labels,) * len(data_fl["spikeTimes"])
        else:
            data_fl["neur_regions"] = (((fl_info["region"],) * n_neur_fl),) * len(
                data_fl["spikeTimes"]
            )
        data_fl_pd = pd.DataFrame.from_dict(data_fl)
        cats = data_fl["stim_sample_MAIN"]
        opp_cats = np.zeros_like(cats)
        opp_cats[cats == 1] = 2
        opp_cats[cats == 2] = 1
        corr_targ = data_fl["targ_cor"]
        cho_targ = data_fl["targ_cho"]
        choices = np.zeros_like(cats)
        choices[:] = np.nan
        mask_corr = corr_targ == cho_targ
        choices[mask_corr] = cats[mask_corr]
        choices[~mask_corr] = opp_cats[~mask_corr]
        data_fl_pd["chosen_cat"] = choices
        cat, anticat = _get_projs(data_fl_pd)
        data_fl_pd["cat_proj"] = cat
        data_fl_pd["anticat_proj"] = anticat

        datas.append(data_fl_pd)
        n_neurs.append(n_neur_fl)
        files_loaded += 1
        if files_loaded > max_files:
            break
    super_dict = dict(
        date=dates,
        animal=monkeys,
        data=datas,
        n_neurs=n_neurs,
        day=days,
        shape=shapes,
    )
    return super_dict


def filter_valid(
    data,
    targ_key="targ_cho",
    cho_key="chosen_cat",
    cat_key="cat_def_MAIN",
    feat_key="stim_feature_MAIN",
):
    mask1 = data[targ_key].one_of((1, 2))
    mask2 = data[cho_key].one_of((1, 2))
    mask3 = list(~np.isnan(cdm) for cdm in data[cat_key])
    mask4 = list(
        list(u.check_list(x) and len(x) == 2 for x in sfm) for sfm in data[feat_key]
    )
    full_mask = mask1.rs_and(mask2).rs_and(mask3).rs_and(mask4)
    return data.mask(full_mask)
