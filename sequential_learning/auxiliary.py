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


CONFIG_PATH = "sequential_learning/config.conf"
cf = u.ConfigParserColor()
cf.read(CONFIG_PATH)

BASEFOLDER = cf.get("DEFAULT", "BASEFOLDER", fallback="../data/sequential_learning/")


ft1 = (
    "A(?P<date>[0-9a-z]+)\\.(?P<region>[A-Z0-9+]+)\\.learn(?P<shape>A[0-9])"
    "\\-d(?P<day>[0-9]+)\\.[0-9]+\\.FIRA\\.mat"
)
ft2 = (
    "A(?P<date>[0-9a-z]+)\\.(?P<shape>A[0-9]+(\\-t)?)\\-d?(?P<day>[0-9]+)"
    "(?P<post>\\.postA[0-9]+)?\\.FIRA\\.LMAN_"
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


shape_sequence = (
    "A2",
    "A3",
    "A4",
    "A3postA4",
    "A5",
    "A4postA5",
    "A3postA5",
    "A6",
    "A5postA6",
    "A7",
    "A6postA7",
    "A8",
    "A9",
    "A9t",
    "A10",
    "A10t",
)


shape_sequence_nopost = (
    "A2",
    "A3",
    "A4",
    "A5",
    "A6",
    "A7",
    "A8",
    "A9",
    "A9t",
    "A10",
    "A10t",
)


def get_boundary_angles(shapes=None, data_folder=BASEFOLDER, **kwargs):
    if shapes is None:
        shapes = shape_sequence
    angle_dict = {}
    for shape in shapes:
        folder = os.path.join(data_folder, shape)
        out = load_kiani_data_folder(folder, max_files=1, **kwargs)
        cd = out["data"][0]["cat_def_MAIN"].iloc[0]
        angle_dict[shape] = cd
    return angle_dict


def get_num_sessions():
    pass


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
    n_bins=3,
    eps=1e-10,
    n_samps=1,
):
    cats = data[cat_proj]
    anti_cats = data[anticat_proj]
    if proj_range is None:
        p_end = np.max(list(np.max(np.abs(x)) for x in cats))
        p_end = np.min([0.71, p_end])
        proj_range = (-p_end - eps, p_end + eps)
    edges = (np.linspace(*proj_range, n_bins + 1),) * 2
    masks = []
    for i, cat_i in enumerate(cats):
        anti_cat_i = anti_cats[i]
        sample_pos = np.stack((cat_i, anti_cat_i), axis=1)
        nan_mask = np.any(
            np.logical_or(sample_pos < proj_range[0], sample_pos > proj_range[-1]),
            axis=1,
        )
        sample_pos_reduced = sample_pos[~nan_mask]
        reduced_inds = np.arange(sample_pos.shape[0])[~nan_mask]

        counts_distrib, edges, binnumber = sts.binned_statistic_dd(
            sample_pos_reduced,
            np.ones(len(sample_pos_reduced)),
            statistic="count",
            bins=edges,
        )
        trl_samps = np.min(counts_distrib)
        if trl_samps > 1:
            splitter = skms.StratifiedShuffleSplit(
                n_splits=n_samps,
                train_size=int(trl_samps) * np.prod(counts_distrib.shape),
            )
            splitter_gen = splitter.split(sample_pos_reduced, binnumber)
            samp_inds = list(reduced_inds[tr] for tr, _ in splitter_gen)
            samp_inds = np.stack(samp_inds, axis=0)
        else:
            samp_inds = np.zeros((n_samps, 0))
        trl_inds = np.arange(len(cat_i))
        mask_i = np.array(list(np.isin(trl_inds, si) for si in samp_inds)).T
        masks.append(np.squeeze(mask_i))
    return masks


def get_shape_folders(use_folder=BASEFOLDER, pattern="A[0-9]+[a-z]?(postA[0-9]+)?"):
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
                np.sin(np.radians(cba)),
                np.cos(np.radians(cba)),
            ]
        )
    )
    cat = feats @ cat_vec
    anticat = feats @ anti_cat_vec
    return cat, anticat


angle_corrections = {
    "A4": -30,
    "A5": 30,
}

def fully_sampled(fvs, bound=.7, eps=.1):
    mask = []
    for fv_i in fvs:
        mins = np.min(fv_i, axis=0)
        c1 = np.all(mins < (bound + bound * eps))
        maxs = np.max(fv_i, axis=0)
        c2 = np.all(maxs > (bound - bound * eps))
        c3 = np.any(np.sqrt(np.sum(fv_i ** 2, axis=1)) < eps * bound)
        mask.append(c1 and c2 and c3)
    return np.array(mask)


def load_kiani_data_folder(
    folder,
    templates=file_templates,
    monkey_name="Z",
    max_files=np.inf,
    angle_corrections=angle_corrections,
    center_pt=(-2, -2),
):
    datas = []
    n_neurs = []
    dates = []
    monkeys = []
    days = []
    shapes = []

    files_loaded = 0
    file_gen = u.load_folder_regex_generator(folder, *templates, load_func=load_file)
    center_pt = np.expand_dims(center_pt, 0)
    for fl, fl_info, data_fl in file_gen:
        dates.append(fl_info["date"])
        shape = fl_info["shape"]
        shape_add = fl_info.get("post", "")
        if shape_add is None:
            shape_add = ""
        shapes.append(shape + str(shape_add))
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
        cats = data_fl["stim_sample_MAIN"]
        opp_cats = np.zeros_like(cats)
        opp_cats[cats == 1] = 2
        opp_cats[cats == 2] = 1
        corr_targ = data_fl["targ_cor"]
        cho_targ = data_fl["targ_cho"]
        if shape in angle_corrections.keys():
            data_fl["cat_def_MAIN"] = data_fl["cat_def_MAIN"] + angle_corrections[shape]
        data_fl_pd = pd.DataFrame.from_dict(data_fl)
        if "stim_aperture" in data_fl.keys():
            pos = np.stack(list(tuple(x) for x in data_fl["stim_aperture"]), axis=0)
            uv = u.make_unit_vector(pos[:, :2] - center_pt)
            ang = np.arctan2(uv[:, 1], uv[:, 0]) - np.arctan2(0, 1)
            rot = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
            feats = np.stack(
                list(tuple(x) for x in data_fl["stim_feature_MAIN"]), axis=0
            )
            cd_vec = u.radian_to_sincos(np.radians(data_fl["cat_def_MAIN"]))
            new_vec = np.array(
                list(tuple(cd_i @ rot[..., i]) for i, cd_i in enumerate(cd_vec))
            )
            new_cat_def = np.degrees(u.sincos_to_radian(new_vec[:, 0], new_vec[:, 1]))

            new_feats = list(tuple(fi @ rot[..., i]) for i, fi in enumerate(feats))
            data_fl_pd["screen_angle"] = ang
            data_fl_pd["stim_feature_screen"] = new_feats
            data_fl_pd["feature_vector"] = list(tuple(x) for x in uv)
            data_fl_pd["cat_def_screen"] = new_cat_def
        else:
            fv = np.zeros((len(cats), 2))
            cd = np.zeros(len(cats))
            fv[:] = np.nan
            data_fl_pd["feature_vector"] = list(tuple(x) for x in fv)
            data_fl_pd["stim_feature_screen"] = list(tuple(x) for x in fv)
            data_fl_pd["cat_def_screen"] = cd
            data_fl_pd["screen_angle"] = cd
            
        choices = np.zeros_like(cats)
        choices[:] = np.nan
        mask_corr = corr_targ == cho_targ
        data_fl_pd["correct"] = mask_corr
        choices[mask_corr] = cats[mask_corr]
        choices[~mask_corr] = opp_cats[~mask_corr]
        data_fl_pd["chosen_cat"] = choices
        cat_proj, anticat_proj = _get_projs(data_fl_pd)
        u_cats = np.unique(cats[cat_proj > 0])
        try:
            assert len(u_cats) == 1
        except AssertionError:
            print(shape)
            print(np.unique(cats[cat_proj > 0], return_counts=True))
            cp1 = cat_proj[cats == 1]
            print(cp1[cp1 > 0])
        if cats[cat_proj > 0][0] == 1:
            cat_proj = -cat_proj
            data_fl_pd["cat_def_MAIN"] = u.normalize_periodic_range(
                data_fl_pd["cat_def_MAIN"] + 180, radians=False,
            )
            print("{} flipped category projection".format(shape))
        data_fl_pd["cat_proj"] = cat_proj
        data_fl_pd["anticat_proj"] = anticat_proj

        datas.append(data_fl_pd)
        n_neurs.append(n_neur_fl)
        files_loaded += 1
        if files_loaded >= max_files:
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


def _filter_and_categorize(
    data,
    *centroids,
    sample_radius=100,
    stim_feat_field="stim_feature_MAIN",
):
    masks = []
    stim_feats = list(np.stack(sf, axis=0) for sf in data[stim_feat_field])
    for cent in centroids:
        sub_masks = []
        if len(cent.shape) == 1:
            cent = np.expand_dims(cent, 0)
        for sf in stim_feats:
            m_sf = np.sqrt(np.sum((sf - cent) ** 2, axis=1)) < sample_radius
            sub_masks.append(m_sf)
        masks.append(sub_masks)
    return masks


def parse_dates(dates):
    dates = list(date[:-1] for date in dates)
    out = pd.to_datetime(dates, yearfirst=True)
    return out


def get_binary_feature_masks(*datas, feat_ind=0, feat_field="stim_feature_MAIN"):
    masks = []
    for data in datas:
        mask1 = list((np.stack(x, axis=0)[:, feat_ind] > 0) for x in data[feat_field])
        mask2 = list((np.stack(x, axis=0)[:, feat_ind] <= 0) for x in data[feat_field])

        masks.append((gio.ResultSequence(mask1), gio.ResultSequence(mask2)))
    return masks


def get_strict_prototype_masks(
    *datas,
    cat_proj_field="cat_proj",
    anticat_proj_field="anticat_proj",
    min_cat_bound=0.42,
    max_cat_bound=0.71,
    ac_bound=0.141,
    single_mask=False,
):
    masks = []
    for data in datas:
        cat = data[cat_proj_field]
        acat = data[anticat_proj_field]
        ac_con = (acat > -ac_bound).rs_and(acat < ac_bound)
        m1 = (cat > min_cat_bound).rs_and(cat < max_cat_bound)
        m1 = m1.rs_and(ac_con)
        m2 = (cat > -max_cat_bound).rs_and(cat < -min_cat_bound)
        m2 = m2.rs_and(ac_con)
        if single_mask:
            ret_m = m1.rs_or(m2)
        else:
            ret_m = (m1, m2)
        masks.append(ret_m)
    return masks


def single_prototype_mask(
        *args, **kwargs,
):
    return get_strict_prototype_masks(*args, single_mask=True, **kwargs)


def get_prototype_masks(
    *datas,
    cat_field="stim_sample_MAIN",
    stim_feat_field="stim_feature_MAIN",
    session_ind=0,
    data_ind=0,
    sample_radius=100,
):
    main_data = datas[data_ind]
    cat1_mask = main_data[cat_field] == 1
    cat1_stim = main_data.mask(cat1_mask)
    c1_arr = np.stack(cat1_stim[stim_feat_field][session_ind], axis=0)
    cat1_average = np.mean(
        c1_arr,
        axis=0,
    )

    cat2_mask = main_data[cat_field] == 2
    cat2_stim = main_data.mask(cat2_mask)
    c2_arr = np.stack(cat2_stim[stim_feat_field][session_ind], axis=0)
    cat2_average = np.mean(
        c2_arr,
        axis=0,
    )

    main_masks = _filter_and_categorize(
        main_data,
        cat1_average,
        cat2_average,
        sample_radius=sample_radius,
    )

    masks = [main_masks]
    for data in datas:
        mask = _filter_and_categorize(
            data,
            cat1_average,
            cat2_average,
            sample_radius=sample_radius,
        )
        masks.append(mask)
    return masks
