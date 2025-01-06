import os
import scipy.io as sio
import re
import numpy as np
import pandas as pd
import scipy.stats as sts
import sklearn.model_selection as skms
from PIL import Image

import general.utility as u
import general.data_io as gio


CONFIG_PATH = "sequential_learning/config.conf"
cf = u.ConfigParserColor()
cf.read(CONFIG_PATH)

ROOTFOLDER = cf.get("DEFAULT", "ROOTFOLDER", fallback="../data/sequential_learning/")
BASEFOLDER = os.path.join(ROOTFOLDER, "data")
FIXATIONFOLDER = os.path.join(ROOTFOLDER, "fixation")
STIMFOLDER = os.path.join(ROOTFOLDER, "stimuli")


ft1 = (
    "A(?P<date>[0-9a-z]+)\\.(?P<region>[A-Z0-9+]+)\\.learn(?P<shape>A[0-9])"
    "\\-d(?P<day>[0-9]+)\\.[0-9]+\\.FIRA\\.mat"
)
ft2 = (
    "A(?P<date>[0-9a-z]+)\\.(?P<shape>A[0-9]+(\\-t)?)\\-d?(?P<day>[0-9]+)"
    "(?P<post>\\.postA[0-9]+)?\\.FIRA\\.LMAN_"
    "cat\\.(?P<region>[A-Z0-9+]+)\\.mat"
)
ft3 = (
    "A(?P<date>[0-9a-z]+)\\.fix(?P<shape>A[0-9]+(\\-t)?)"
    "(?P<post>\\.postA[0-9]+)?\\.(?P<time>(before|after)\\.)?FIRA\\.fixation"
    "\\.(?P<region>[A-Z0-9+]+)\\.mat"
)
file_templates = (ft1, ft2, ft3)

default_type_dict = {
    "time": float,
    "value": float,
    "task_ver": object,
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


shape_sequence_unique = shape_sequence_nopost[:-3]


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


def merge_task_info_with_passive(passive, task):
    task[""]


def load_shape_list(
    shapes,
    data_folder=BASEFOLDER,
    fixation_data_folder=FIXATIONFOLDER,
    sort_by="day",
    max_files=np.inf,
    exclude_invalid=True,
    with_passive=False,
    **kwargs,
):
    out_data = {}
    passive_data = {}
    for shape in shapes:
        data = gio.Dataset.from_readfunc(
            load_kiani_data_folder,
            os.path.join(data_folder, shape),
            max_files=max_files,
            sort_by=sort_by,
            **kwargs,
        )
        if exclude_invalid:
            data = filter_valid(data)
        out_data[shape] = data
        if with_passive:
            angs = np.concatenate(data["cat_def_MAIN"])
            extra_fixation_info = {
                "cat_def_MAIN": get_common_angle(angs),
            }
            data_passive = gio.Dataset.from_readfunc(
                load_kiani_data_folder,
                os.path.join(fixation_data_folder, shape),
                max_files=max_files,
                sort_by=sort_by,
                fixation=True,
                extra_fixation_info=extra_fixation_info,
            )
            task_dates = parse_dates(data["date"])
            task_days = data["day"]
            fix_dates = parse_dates(data_passive["date"])
            fix_days = match_dates(fix_dates, task_dates, task_days)
            data_passive.data["day"] = fix_days
            data_passive = data_passive.resort_sessions("day")
            print(data_passive["day"])
            passive_data[shape] = data_passive
    if with_passive:
        out = (out_data, passive_data)
    else:
        out = out_data
    return out


def _get_ith_feature(feats, i, shape=2):
    feats_all = []
    orders_all = []
    for feat in feats:
        feats_sess = []
        for x in feat:
            if u.check_list(x) and len(x) > i:
                feats_sess.append(np.array(x[i]).reshape((shape,)))
            else:
                feats_sess.append(np.array((np.nan,) * shape))
        orders_all.append(np.ones(len(feats_sess)) * i)
        feats_sess = np.array(feats_sess).reshape((-1, shape))
        feats_all.append(feats_sess)
    return feats_all, orders_all


def get_fixation_stim_responses(
    data,
    tbeg=50,
    twid=300,
    stim_on_templ="stim_on_{}",
    max_stim=8,
    feature_fields=("stim_feature_MAIN", "cat_proj", "anticat_proj", "shapes"),
    feature_shapes=(2, 1, 1, 1),
    shown_stim_field="shown_stim_num",
    filter_nan=True,
    **kwargs,
):
    stim_on_fields = list(stim_on_templ.format(i) for i in range(1, max_stim + 1))
    pops_all, orders_all = [], []
    feats_out = {}
    for i, sof in enumerate(stim_on_fields):
        mask = data[sof].rs_isnan().rs_not()
        mask = mask.rs_and(data[shown_stim_field] >= i)
        data_i = data.mask(mask)
        pops, xs = data_i.get_neural_activity(
            twid, tbeg, tbeg, twid, time_zero_field=sof, **kwargs
        )
        for j, ff in enumerate(feature_fields):
            feats_ff, orders = _get_ith_feature(data_i[ff], i, shape=feature_shapes[j])
            feats_out_ff = feats_out.get(ff, [])
            feats_out_ff.append(feats_ff)
            feats_out[ff] = feats_out_ff
        n_sessions = len(pops)
        pops_all.append(pops)
        orders_all.append(orders)
    pop_groups_all = []
    feats_combined_all = {}
    order_groups_all = []
    for i in range(n_sessions):
        pops_group = np.concatenate(
            list(p[i] for p in pops_all if p[i].shape[0] > 0), axis=0
        )
        feats_combined = {}
        for ff in feature_fields:
            feats_all = feats_out[ff]
            feats_group = np.concatenate(list(f[i] for f in feats_all), axis=0)
            feats_combined[ff] = feats_group
        orders_group = np.concatenate(list(f[i] for f in orders_all), axis=0)
        if filter_nan:
            mask = ~np.any(np.isnan(feats_combined[feature_fields[0]]), axis=1)
            pops_group = pops_group[mask]
            feats_combined = {
                ff: feats_group[mask] for ff, feats_group in feats_combined.items()
            }
            feats_group = feats_group[mask]
            orders_group = orders_group[mask]
        pop_groups_all.append(pops_group)
        for ff in feature_fields:
            x = feats_combined_all.get(ff, [])
            x.append(feats_combined[ff])
            feats_combined_all[ff] = x
        order_groups_all.append(orders_group)

    return pop_groups_all, feats_combined_all, order_groups_all


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


def load_all_images(image_seq=shape_sequence_unique, basefolder=STIMFOLDER, **kwargs):
    imgs = {}
    for img in image_seq:
        path = os.path.join(basefolder, img)
        imgs[img] = load_stim_images(path, **kwargs)
    return imgs


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
            img = np.array(Image.open(fp).convert("RGB")) / 255
            img = np.moveaxis(img, -1, 0)
            lv1s.append(lv1)
            lv2s.append(lv2)
            imgs.append(img)
    return np.array(lv1s), np.array(lv2s), np.stack(imgs, axis=0)


feat_pattern = "(?P<shape>A[0-9])/(?P<f1>[\\-0-9]+)_(?P<f2>[\\-0-9]+)_.*"


def _extract_feats(s, pattern=feat_pattern):
    m = re.match(pattern, s[0])
    if m is None:
        shape = np.nan
        f1 = np.nan
        f2 = np.nan
    else:
        shape = m.group("shape")
        f1 = float(m.group("f1"))
        f2 = float(m.group("f2"))
    return shape, (f1, f2)


def _extract_feat_list(img_strs, **kwargs):
    shapes = []
    feats = []
    for s in img_strs:
        shape_i, feat_i = _extract_feats(s, **kwargs)
        shapes.append(shape_i)
        feats.append(feat_i)
    return np.array(shapes), np.stack(feats, axis=0)


def process_fixation(full_dict):
    imgs = full_dict["imagename"]
    num_stim = full_dict["shown_stim_num"]
    feats = np.zeros(len(imgs), dtype=object)
    shapes = np.zeros_like(feats)
    for i, trl_img in enumerate(imgs):
        img_strs = trl_img[: int(num_stim[i])]

        if len(img_strs) > 0:
            shapes[i], feats[i] = _extract_feat_list(img_strs)
        else:
            shapes[i] = np.nan
            feats[i] = np.nan
    full_dict["shapes"] = shapes
    full_dict["stim_feature_MAIN"] = feats
    return full_dict


def load_file(fname, type_dict=default_type_dict, fixation=True):
    data = sio.loadmat(fname)
    fira = data["FIRA"]
    keys, types = fira[0, 0]["events"][0, 0]
    vals = fira[0, 1]
    full_dict = {}
    for i, k in enumerate(keys):
        if u.check_list(k):
            k = k[0]
        vals_i = vals[:, i]
        if (t := type_dict.get(k)) is not None:
            use_type = t
        else:
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
                except ValueError as e:
                    print(k, use_type, vuj)
                    print(types[i][0])
                    print(e)
                    vals_use[j] = np.nan
        full_dict[k] = vals_use
    spike_times_all = fira[0, 2][:, 0]
    spks = np.zeros(
        (spike_times_all.shape[0], spike_times_all[0].shape[0]), dtype=object
    )
    for i, spk_trl in enumerate(spike_times_all):
        spks[i] = list(np.squeeze(neur_trl) for neur_trl in spk_trl[:, 0])
    full_dict["spikeTimes"] = list(s for s in spks)
    if fixation:
        full_dict = process_fixation(full_dict)
    return full_dict


def project_on_angle(feats, cba):
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


def get_common_angle(angs):
    cat_boundary_angles, counts = np.unique(angs, return_counts=True)
    cba = cat_boundary_angles[np.argmax(counts)]
    return cba


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
    cba = get_common_angle(
        data[cat_bound_field].to_numpy(),
    )

    return project_on_angle(feats, cba)


angle_corrections = {
    "A4": -30,
    "A5": 30,
}


def fully_sampled(fvs, bound=0.7, eps=0.1):
    mask = []
    for fv_i in fvs:
        mins = np.min(fv_i, axis=0)
        c1 = np.all(mins < (bound + bound * eps))
        maxs = np.max(fv_i, axis=0)
        c2 = np.all(maxs > (bound - bound * eps))
        c3 = np.any(np.sqrt(np.sum(fv_i**2, axis=1)) < eps * bound)
        mask.append(c1 and c2 and c3)
    return np.array(mask)


def process_categorization(data_fl_pd, shape, center_pt):
    cats = data_fl_pd["stim_sample_MAIN"]
    opp_cats = np.zeros_like(cats)
    opp_cats[cats == 1] = 2
    opp_cats[cats == 2] = 1
    corr_targ = data_fl_pd["targ_cor"]
    cho_targ = data_fl_pd["targ_cho"]
    if shape in angle_corrections.keys():
        data_fl_pd["cat_def_MAIN"] = (
            data_fl_pd["cat_def_MAIN"] + angle_corrections[shape]
        )
    if "stim_aperture" in data_fl_pd.columns:
        pos = np.stack(list(tuple(x) for x in data_fl_pd["stim_aperture"]), axis=0)
        uv = u.make_unit_vector(pos[:, :2] - center_pt)
        ang = np.arctan2(uv[:, 1], uv[:, 0]) - np.arctan2(0, 1)
        rot = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
        feats = np.stack(
            list(tuple(x) for x in data_fl_pd["stim_feature_MAIN"]), axis=0
        )
        cd_vec = u.radian_to_sincos(np.radians(data_fl_pd["cat_def_MAIN"]))
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
    if cats[cat_proj > 0].iloc[0] == 1:
        cat_proj = -cat_proj
        data_fl_pd["cat_def_MAIN"] = u.normalize_periodic_range(
            data_fl_pd["cat_def_MAIN"] + 180,
            radians=False,
        )
        print("{} flipped category projection".format(shape))
    data_fl_pd["cat_proj"] = cat_proj
    data_fl_pd["anticat_proj"] = anticat_proj
    return data_fl_pd


def _get_projs_fixation(data, cat_def, feat_key="stim_feature_MAIN"):
    feats = data[feat_key].to_numpy()
    cat_proj = np.zeros(len(feats), dtype=object)
    anticat_proj = np.zeros_like(cat_proj)
    for i, feat_i in enumerate(feats):
        if np.all(np.isnan(feat_i)):
            cat_proj[i] = np.nan
            anticat_proj[i] = np.nan
        else:
            feat_i = feat_i / 1000
            cat_proj[i], anticat_proj[i] = project_on_angle(feat_i, cat_def)
    return cat_proj, anticat_proj


def merge_extra_fixation_info(data, info):
    if (cat_def := info.get("cat_def_MAIN")) is not None:
        cat_proj, anticat_proj = _get_projs_fixation(data, cat_def)
        data["cat_proj"] = cat_proj
        data["anticat_proj"] = anticat_proj
    return data


def load_kiani_data_folder(
    folder,
    templates=file_templates,
    monkey_name="Z",
    max_files=np.inf,
    angle_corrections=angle_corrections,
    center_pt=(-2, -2),
    fixation=False,
    extra_fixation_info=None,
):
    datas = []
    n_neurs = []
    dates = []
    monkeys = []
    days = []
    shapes = []

    files_loaded = 0
    file_gen = u.load_folder_regex_generator(
        folder,
        *templates,
        load_func=load_file,
        fixation=fixation,
    )
    center_pt = np.expand_dims(center_pt, 0)
    for fl, fl_info, data_fl in file_gen:
        dates.append(fl_info["date"])
        shape = fl_info["shape"]
        shape_add = fl_info.get("post", "")
        if shape_add is None:
            shape_add = ""
        shapes.append(shape + str(shape_add))
        day_fl = fl_info.get("day", None)
        if day_fl is not None:
            day_fl = int(day_fl)
        days.append(day_fl)
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
        if not fixation:
            data_fl_pd = process_categorization(data_fl_pd, shape, center_pt)
        if fixation and extra_fixation_info is not None:
            data_fl_pd = merge_extra_fixation_info(data_fl_pd, extra_fixation_info)

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


def match_dates(d1, d2, c2):
    out = []
    for d1_i in d1:
        mask = d1_i == d2
        if np.any(mask):
            out.append(c2[np.where(mask)[0][0]])
        else:
            if np.all(d1_i < d2):
                out.append(0)
            elif np.all(d1_i > d2):
                out.append(500)
            else:
                out.append(None)
    return out


def get_binary_feature_masks(*datas, feat_ind=0, feat_field="stim_feature_MAIN"):
    masks = []
    for data in datas:
        mask1 = list((np.stack(x, axis=0)[:, feat_ind] > 0) for x in data[feat_field])
        mask2 = list((np.stack(x, axis=0)[:, feat_ind] <= 0) for x in data[feat_field])

        masks.append((gio.ResultSequence(mask1), gio.ResultSequence(mask2)))
    return masks


def proto_box_mask(fields, c1="cat_proj", c2="anticat_proj"):
    return strict_prototype_mask_cat_acat(fields[c1], fields[c2])


def strict_prototype_mask_cat_acat(
    cat,
    acat,
):
    cat = gio.ResultSequence(pd.Series(np.squeeze(x)) for x in cat)
    acat = gio.ResultSequence(pd.Series(np.squeeze(x)) for x in acat)
    return _box_mask(cat, acat, single_mask=True)


BOX_MIN = 0.42
BOX_MAX = 0.71
BOX_AC = 0.1415
def box_mask_array(
        arr,
        min_cat_bound=BOX_MIN,
        max_cat_bound=BOX_MAX,
        ac_bound=BOX_AC,
        single_mask=True,
        cat_ind=0,
        acat_ind=1,
):
    cat, acat = arr[:, cat_ind], arr[:, acat_ind]
    ac_con = np.logical_and(acat > -ac_bound , acat < ac_bound)
    m1 = np.logical_and(cat > min_cat_bound , cat < max_cat_bound)
    m1 = np.logical_and(m1, ac_con)
    m2 = np.logical_and(cat < -min_cat_bound, cat > -max_cat_bound)
    m2 = np.logical_and(m2, ac_con)
    if single_mask:
        ret_m = np.logical_or(m1, m2)
    else:
        ret_m = (m1, m2)
    return ret_m


def _box_mask(
    cat,
    acat,
    min_cat_bound=BOX_MIN,
    max_cat_bound=BOX_MAX,
    ac_bound=BOX_AC,
    single_mask=False,
):
    ac_con = (acat > -ac_bound).rs_and(acat < ac_bound)
    m1 = (cat > min_cat_bound).rs_and(cat < max_cat_bound)
    m1 = m1.rs_and(ac_con)
    m2 = (cat > -max_cat_bound).rs_and(cat < -min_cat_bound)
    m2 = m2.rs_and(ac_con)
    if single_mask:
        ret_m = m1.rs_or(m2)
    else:
        ret_m = (m1, m2)
    return ret_m


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
        ret_m = _box_mask(cat, acat)
        masks.append(ret_m)
    return masks


def single_prototype_mask(
    data,
    **kwargs,
):
    return get_strict_prototype_masks(data, single_mask=True, **kwargs)[0]


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
