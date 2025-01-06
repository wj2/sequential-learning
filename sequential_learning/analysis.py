import itertools as it
import numpy as np
import scipy.stats as sts
import sklearn.svm as skm
import sklearn.gaussian_process as skgp
import sklearn.linear_model as sklm
import sklearn.metrics.pairwise as skmp

import general.utility as u
import general.neural_analysis as na

import sequential_learning.auxiliary as slaux


def get_shape_resps(
    data,
    time_zero_field="stim_on",
    tbeg=-500,
    tend=0,
    winsize=500,
    regions=("IT",),
    min_neurs=1,
    min_trls=10,
):
    out_dict = {}
    for k, v in data.items():
        pops, xs = v.get_neural_activity(
            winsize,
            tbeg,
            tend,
            time_zero_field=time_zero_field,
            regions=regions,
            skl_axes=True,
        )
        pops_save = []
        for pop in pops:
            if pop.shape[0] > min_neurs and pop.shape[2] > min_trls:
                channel_mask = np.squeeze(np.any(np.std(pop, axis=2) > 0, axis=-1))
                pops_save.append(pop[channel_mask])
        out_dict[k] = pops_save, xs
    return out_dict


def uniform_sample_mask(data, **kwargs):
    mask = slaux.sample_uniform_mask(data, **kwargs)
    data = data.mask(mask)
    session_mask = list(np.sum(x) > 0 for x in mask)
    data = data.session_mask(session_mask)
    return data


def find_positions(data_dict, pos_key="stim_aperture"):
    pos_dict = {}
    for shape, shape_data in data_dict.items():
        if pos_key in shape_data.session_keys:
            pos_cat = np.concatenate(shape_data[pos_key])
            u_pos = np.unique(list(tuple(x) for x in pos_cat), axis=0)
            pos_dict[shape] = u_pos

    return pos_dict


def stack_features(feats, ind=None, div=1000):
    stacked_feats = []
    if ind is not None:
        feats = (feats[ind],)
    for i, feat in enumerate(feats):
        stacked_feats.append(np.stack(feat.to_numpy(), axis=0) / div)
    if ind is not None:
        stacked_feats = stacked_feats[0]
    return stacked_feats


def compute_unit_dprime(
    data,
    *args,
    cat_field="stim_sample_MAIN",
    time_zero_field="stim_on",
    day_field="day",
    cho_field="chosen_cat",
    targ_field="stim_sample_MAIN",
    uniform_resample=False,
    **kwargs,
):
    if uniform_resample:
        data = uniform_sample_mask(data)
    targs = data[cat_field]
    t1_mask = targs == 1
    t2_mask = targs == 2
    xs, (pop1, pop2) = data.get_dec_pops(
        *args,
        t1_mask,
        t2_mask,
        tzfs=(time_zero_field, time_zero_field),
        shuffle_trials=False,
        **kwargs,
    )
    days = data[day_field]
    dprimes = np.zeros((len(days), len(xs)))
    performance = np.zeros(len(days))
    for i, p1_i in enumerate(pop1):
        p2_i = pop2[i]
        if p1_i.shape[0] > 0:
            mu1 = np.squeeze(np.mean(p1_i, axis=2))
            sig1 = np.squeeze(np.std(p1_i, axis=2))
            mu2 = np.squeeze(np.mean(p2_i, axis=2))
            sig2 = np.squeeze(np.std(p2_i, axis=2))
            dprimes[i] = np.nanmean(np.abs(mu1 - mu2) / np.sqrt(sig1 * sig2), axis=0)
            corr = data[cho_field][i] == data[targ_field][i]
            performance[i] = np.nanmean(corr)
        else:
            dprimes[i] = np.nan
            performance[i] = np.nan
    return xs, dprimes, performance, days


def compute_cross_shape_generalization(
    pre_data,
    post_data,
    *args,
    stim_feat_field="stim_feature_MAIN",
    day_field="day",
    cat_field="stim_sample_MAIN",
    date_field="date",
    sample_radius=300,
    **kwargs,
):
    pre_dates = slaux.parse_dates(pre_data[date_field])
    post_dates = slaux.parse_dates(post_data[date_field])
    date_diff = post_dates[0] - pre_dates[-1]

    pre_targ_diff = pre_dates[-1] - pre_dates - date_diff
    pre_ind = np.argmin(np.abs(pre_targ_diff))
    if pre_targ_diff[pre_ind].days > 0:
        print(
            "desired date difference is not exact, is {}".format(pre_targ_diff[pre_ind])
        )
    d1 = pre_dates[pre_ind]
    d2 = pre_dates[-1]
    pre_mask = np.logical_or(pre_dates == d1, pre_dates == d2)
    data_pre = pre_data.session_mask(pre_mask)

    post_targ_diff = post_dates[0] - post_dates + date_diff
    post_ind = np.argmin(np.abs(post_targ_diff))
    if post_targ_diff[post_ind].days > 0:
        print(
            "desired date difference is not exact, is {}".format(
                post_targ_diff[post_ind]
            )
        )
    d1 = post_dates[0]
    d2 = post_dates[post_ind]
    post_mask = np.logical_or(post_dates == d1, post_dates == d2)
    data_post = post_data.session_mask(post_mask)

    # post data has most restricted set of points
    post_dec_masks, pre_dec_masks = slaux.get_prototype_masks(data_post, data_pre)

    ind_pairs = ((-1, 0),)
    out_shape = cross_data_generalization(
        data_pre,
        pre_dec_masks,
        data_post,
        post_dec_masks,
        *args,
        ind_pairs=ind_pairs,
        **kwargs,
    )
    ind_pairs = ((0, 1),)
    out_session_pre = cross_data_generalization(
        data_pre,
        pre_dec_masks,
        data_pre,
        pre_dec_masks,
        *args,
        ind_pairs=ind_pairs,
        **kwargs,
    )
    out_session_post = cross_data_generalization(
        data_post,
        post_dec_masks,
        data_post,
        post_dec_masks,
        *args,
        ind_pairs=ind_pairs,
        **kwargs,
    )
    return out_shape, out_session_pre, out_session_post


def estimate_nonlinear_decision_boundary(
    data,
    ind=0,
    chosen_category_field="chosen_cat",
    stim_feat_field="stim_feature_MAIN",
):
    cats_ch = data[chosen_category_field][ind].to_numpy(int)
    feats = stack_features(data[stim_feat_field], ind=ind)

    m = skm.SVC()
    m.fit(feats, cats_ch)
    coherence = m.score(feats, cats_ch)
    out = {
        "coherence": coherence,
        "model": m,
    }
    return out


def estimate_decision_boundary(
    data,
    ind=0,
    chosen_category_field="chosen_cat",
    stim_feat_field="stim_feature_MAIN",
):
    cats_ch = data[chosen_category_field][ind].to_numpy(int)
    feats = stack_features(data[stim_feat_field], ind=ind)

    m = skm.LinearSVC()
    m.fit(feats, cats_ch)
    coherence = m.score(feats, cats_ch)
    coef = u.make_unit_vector(m.coef_)
    degree = np.degrees(np.arccos(u.make_unit_vector(np.array([-1, 1])) @ coef))
    inter = m.intercept_
    out = {"coherence": coherence, "boundary": (coef, inter), "degree": degree}
    return out


def get_feature_responses(
    data,
    begin,
    end,
    time_zero_field="stim_on",
    feat_field="stim_feature_MAIN",
    **kwargs,
):
    pop_out = data.get_populations(
        end - begin,
        begin,
        end,
        time_zero_field=time_zero_field,
        **kwargs,
    )
    pops, xs = pop_out[:2]
    targ_x = (begin + end) / 2
    pops_t = []
    x_ind = np.argmin((xs - targ_x) ** 2)
    for i, pop in enumerate(pops):
        pops_t.append(np.squeeze(pop[..., x_ind]))

    feats = stack_features(data[feat_field])
    return feats, pops_t


def estimate_feature_manifold(feats, resps, kernel=None):
    m = skgp.GaussianProcessRegressor(kernel=kernel)
    m.fit(feats, resps)
    return m


def bootstrap_dimensionality(pop, n=1000, use_feats=None):
    if use_feats is not None:
        m = estimate_feature_manifold(use_feats, pop)
        pop = m.predict(use_feats)
    out = u.bootstrap_list(pop, u.participation_ratio, n=n)
    return out


def _make_out_dict(
    data,
    out,
    day_field="day",
    shape_field="shape",
    region_field="neur_regions",
    **kwargs,
):
    days = data[day_field]
    shapes = data[shape_field]
    regions = list(drf.iloc[0][0] for drf in data[region_field])

    out_dict = {}
    for i, dec in enumerate(out[0]):
        key = (days[i], shapes[i], regions[i])
        out_dict[key] = tuple(x[i] for x in out[0:1] + out[2:])

    xs = out[1]
    out_full = (out_dict, xs)
    return out_full


def _format_cross_session_info(
    *dm_pairs,
    winsize=500,
    tbeg=-500,
    tend=500,
    shape_field="shape",
    date_field="date",
    region="IT",
    time_zero_field="stim_on",
    stepsize=50,
):
    pops1 = []
    pops2 = []
    shapes = []
    dates = []
    for data, masks in dm_pairs:
        xs, (pop1, pop2) = data.get_dec_pops(
            winsize,
            tbeg,
            tend,
            stepsize,
            *masks,
            tzfs=(time_zero_field,) * len(masks),
            regions=(region,),
            shuffle_trials=False,
        )
        mask = []
        for i, pop1_i in enumerate(pop1):
            pop2_i = pop2[i]
            mask_i = pop1_i.shape[0] > 0 and pop1_i.shape[2] > 1 and pop2_i.shape[2] > 1
            mask.append(mask_i)
        mask = np.array(mask)
        shapes.extend(np.array(data[shape_field])[mask])
        dates.extend(np.array(data[date_field])[mask])
        pop1 = np.array(pop1, dtype=object)[mask]
        pop2 = np.array(pop2, dtype=object)[mask]
        pops1.extend(pop1)
        pops2.extend(pop2)
    return shapes, dates, xs, (pops1, pops2)


def _generalize_cross_session_decoder(
    shapes,
    dates,
    pops1,
    pops2,
    indiv_zscore=True,
    n_folds=20,
    model=skm.LinearSVC,
    **kwargs,
):
    if indiv_zscore:
        for i, p1_i in enumerate(pops1):
            p2_i = pops2[i]
            p1_i, p2_i = na.zscore_tc(p1_i, p2_i)
            pops1[i] = p1_i
            pops2[i] = p2_i
        kwargs["norm"] = False

    out_gen = np.zeros((len(pops1), len(pops2)), dtype=object)
    out_var = np.zeros((len(pops1), len(pops2)), dtype=object)
    for i, p1_i in enumerate(pops1):
        p2_i = pops2[i]
        out = na.fold_skl(
            p1_i,
            p2_i,
            n_folds,
            mean=False,
            model=model,
            return_projection=True,
            **kwargs,
        )
        out_gen[i, i] = out["score"]
        out_var[i, i] = np.var(out["projection"], axis=-1)

        for j, p1_j in enumerate(pops1):
            p2_j = pops2[j]
            if i != j:
                out_gen[i, j] = na.apply_estimators_discrete(
                    out["estimators"],
                    p1_j,
                    p2_j,
                )
                var_ij = na.project_on_estimators_discrete(
                    out["estimators"],
                    p1_j,
                    p2_j,
                )
                out_var[i, j] = np.var(var_ij, axis=-1)
    return shapes, dates, out_gen, out_var


def choice_projection_tc(
    s1,
    s2,
    t_ind=0,
    s1_ind=-1,
    s1_dv_ind=-2,
    s2_ind=0,
    s2_dv_ind=1,
    dv_ind=0,
    choice_field="chosen_cat",
    cat_field="stim_sample_MAIN",
    test_trls=100,
    avg_wid=20,
    both_start=False,
    use_all_dv_ind=False,
):
    p_s1 = s1["pops"][s1_ind][..., t_ind]
    p_s2 = s2["pops"][s2_ind][..., t_ind]
    if use_all_dv_ind:
        dv_s1 = s1["dvs"][s1_dv_ind][..., t_ind]
        dv_s1 = np.concatenate(list(dv_s1[..., j] for j in range(dv_s1.shape[-1])))
        dv_s2 = s1["dvs"][s2_dv_ind][..., t_ind]
        dv_s2 = np.concatenate(list(dv_s2[..., j] for j in range(dv_s2.shape[-1])))
    else:
        dv_s1 = s1["dvs"][s1_dv_ind][..., dv_ind, t_ind]
        dv_s2 = s2["dvs"][s2_dv_ind][..., dv_ind, t_ind]
    c_s1 = s1["trial_info"][s1_ind][choice_field]
    c_s2 = s2["trial_info"][s2_ind][choice_field]
    t_s1 = s1["trial_info"][s1_ind][cat_field]
    t_s2 = s2["trial_info"][s2_ind][cat_field]

    projs_s1_to_s1 = (dv_s1 @ p_s1).T
    projs_s2_to_s1 = (dv_s1 @ p_s2).T
    projs_s1_to_s2 = (dv_s2 @ p_s1).T
    projs_s2_to_s2 = (dv_s2 @ p_s2).T

    corr_traj = {}
    m_s1 = sklm.LogisticRegression()
    if both_start:
        proj_tr, c_tr = projs_s1_to_s1[test_trls:], c_s1[test_trls:]
        proj_te_s1, c_te_s1 = projs_s1_to_s1[:test_trls], c_s1[:test_trls]
        proj_te_s2, c_te_s2 = projs_s2_to_s1[:test_trls], c_s1[:test_trls]
        t_s1_te = t_s1[:test_trls]
    else:
        proj_tr, c_tr = projs_s1_to_s1[:-test_trls], c_s1[:-test_trls]
        proj_te_s1, c_te_s1 = projs_s1_to_s1[-test_trls:], c_s1[-test_trls:]
        proj_te_s2, c_te_s2 = projs_s2_to_s1[:test_trls], c_s1[:test_trls]
        t_s1_te = t_s1[-test_trls:]
    m_s1.fit(proj_tr, c_tr)
    preds_s1_on_s1 = m_s1.predict(proj_te_s1)
    preds_s2_on_s1 = m_s1.predict(proj_te_s2)
    corr_traj["s1 on s1"] = s1 = preds_s1_on_s1 == c_te_s1
    corr_traj["s2 on s1"] = preds_s2_on_s1 == c_te_s2
    corr_traj["s1 corr"] = t_s1_te == c_te_s1

    m_s2 = sklm.LogisticRegression()
    proj_tr, c_tr = projs_s2_to_s2[test_trls:], c_s2[test_trls:]
    if both_start:
        proj_te_s1, c_te_s1 = projs_s1_to_s2[:test_trls], c_s1[:test_trls]
    else:
        proj_te_s1, c_te_s1 = projs_s1_to_s2[-test_trls:], c_s1[-test_trls:]
    proj_te_s2, c_te_s2 = projs_s2_to_s2[:test_trls], c_s2[:test_trls]
    m_s2.fit(proj_tr, c_tr)
    preds_s1_on_s2 = m_s2.predict(proj_te_s1)
    preds_s2_on_s2 = m_s2.predict(proj_te_s2)
    corr_traj["s1 on s2"] = preds_s1_on_s2 == c_te_s1
    corr_traj["s2 on s2"] = preds_s2_on_s2 == c_te_s2
    corr_traj["s2 corr"] = t_s2[:test_trls] == c_te_s2

    smooth_corr_traj = {}
    filt = np.ones(avg_wid) / avg_wid
    for k, traj in corr_traj.items():
        smooth_corr_traj[k] = np.convolve(traj, filt, mode="valid")
    return smooth_corr_traj


def _uniform_labels(
    data, fields=("cat_proj", "anticat_proj"), min_chamber=10, range_=(-1, 1), n_bins=3
):
    group = data[list(fields)]
    out_labels = []
    out_pass = []
    for g in group:
        bin_counts, _, labels = sts.binned_statistic_dd(
            g.to_numpy(),
            np.ones(len(g)),
            statistic="sum",
            range=(range_,) * len(fields),
            bins=n_bins,
        )
        out_pass.append(np.all(bin_counts >= min_chamber))
        out_labels.append(labels)
    return out_labels, out_pass


def joint_variable_shape_sequence(
    data_dict,
    shapes=None,
    t_start=0,
    t_end=500,
    binsize=500,
    binstep=500,
    regions=("IT",),
    time_zero_field="stim_on",
    stim_field=("chosen_cat", "cat_proj", "anticat_proj"),
    keep_session_info=("date",),
    uniform_resample=False,
    uniform_kwargs=None,
    balance_field=None,
    keep_trial_info=("chosen_cat", "stim_sample_MAIN"),
    **kwargs,
):
    if shapes is None:
        shapes = data_dict.keys()
    out_dict = {}
    for shape in shapes:
        data_use = data_dict[shape]
        if uniform_resample:
            if uniform_kwargs is None:
                uniform_kwargs = {}
            kwargs["balance_rel_fields"] = True
            labels, passing = _uniform_labels(data_use, **uniform_kwargs)
            data_use = data_use.session_mask(passing)
            labels = list(label for i, label in enumerate(labels) if passing[i])
            kwargs["rel_flat_all"] = labels
        pops, xs = data_use.get_neural_activity(
            binsize,
            t_start,
            t_end,
            binstep,
            skl_axes=True,
            time_zero_field=time_zero_field,
            regions=regions,
        )
        if u.check_list(stim_field):
            feats = list(np.array(x) for x in data_use[stim_field])
        else:
            feats = list(np.stack(x, axis=0) for x in data_use[stim_field])
        session_info = data_use[list(keep_session_info)].to_numpy()
        trial_info = data_use[list(keep_trial_info)]
        session_dict = joint_variable_decoder(
            pops,
            feats,
            shape_labels=session_info,
            keep_trial_info=trial_info,
            **kwargs,
        )
        out_dict[shape] = session_dict
    return out_dict


def combined_choice_decoder(
    data_dict,
    shapes=None,
    t_start=0,
    t_end=0,
    binsize=500,
    binstep=500,
    regions=("IT",),
    time_zero_field="stim_on",
    stim_field=("chosen_cat", "cat_proj", "anticat_proj"),
    keep_session_info=("date",),
    uniform_resample=False,
    uniform_kwargs=None,
    balance_field=None,
    keep_trial_info=("chosen_cat", "stim_sample_MAIN", "stim_feature_MAIN"),
    last_n_sessions=1,
    **kwargs,
):
    if shapes is None:
        shapes = list(data_dict.keys())
    out_dict = {}
    pre_pops = []
    pre_feats = []
    pre_session_info = []
    pre_trial_info = []

    labels_all = []
    for shape in shapes:
        data_use = data_dict[shape]
        use_days = np.unique(data_use["day"])[-last_n_sessions:]
        data_use = data_use.session_mask(np.isin(data_use["day"], use_days))
        if uniform_resample:
            if uniform_kwargs is None:
                uniform_kwargs = {}
            labels, passing = _uniform_labels(data_use, **uniform_kwargs)
            data_use = data_use.session_mask(passing)
            labels = list(label for i, label in enumerate(labels) if passing[i])
        pops, xs = data_use.get_neural_activity(
            binsize,
            t_start,
            t_end,
            binstep,
            skl_axes=True,
            time_zero_field=time_zero_field,
            regions=regions,
        )
        feats = data_use[list(stim_field)]
        session_info = data_use[list(keep_session_info)]
        trial_info = data_use[list(keep_trial_info)]
        for i, pop in enumerate(pops):
            if pop.shape[0] > 0:
                pre_pops.append(na.zscore_tc(pop)[0])
                pre_feats.append(feats[i])
                pre_session_info.append(session_info.iloc[i])
                pre_trial_info.append(trial_info[i])
                if uniform_resample:
                    labels_all.append(labels[i])
    shapes_comb = "-".join(shapes)
    pre_pops_comb = np.concatenate(pre_pops, axis=2)
    pre_feats_comb = np.concatenate(pre_feats, axis=0)
    pre_trial_info_comb = np.concatenate(pre_trial_info, axis=0)
    if uniform_resample:
        kwargs["balance_rel_fields"] = True
        kwargs["rel_flat_all"] = np.concatenate(labels_all, axis=0)
    session_dict = joint_variable_decoder(
        [pre_pops_comb],
        [pre_feats_comb],
        keep_trial_info=pre_trial_info_comb,
        indiv_zscore=False,
        **kwargs,
    )
    out_dict[shapes_comb] = session_dict
    return out_dict


def _combine_days_fixation(
    pops,
    feats,
    orders,
    days,
    bhv_days=None,
    bhv_feats=None,
    zscore=True,
):
    keep_pops, keep_orders = [], []
    keep_feats = {f: [] for f in feats.keys()}
    keep_bhv_feats = []
    for i, pop in enumerate(pops):
        if pop.shape[1] > 0 and (bhv_days is None or days[i] in bhv_days):
            if zscore:
                pop = na.zscore_tc_shape(pop)
            keep_pops.append(pop)
            list(keep_feats[f].append(feats[f][i]) for f in feats.keys())
            keep_orders.append(orders[i])
            if bhv_feats is not None:
                ind = np.where(bhv_days == days[i])[0][0]
                keep_bhv_feats.append(bhv_feats[ind])
    pops = [np.concatenate(keep_pops, axis=0)]
    feats = {f: [np.concatenate(keep_feats[f], axis=0)] for f in feats.keys()}
    orders = [np.concatenate(keep_orders, axis=0)]
    if bhv_feats is not None:
        bhv_feats = [np.concatenate(keep_bhv_feats, axis=0)]
        bhv_days = np.array([0])
    days = np.array([0])
    return pops, feats, orders, days, bhv_days, bhv_feats


def fixation_generalization_pattern(
    data,
    dec_field="cat_proj",
    gen_field=None,
    gen_func=slaux.proto_box_mask,
    t_start=50,
    binsize=300,
    regions=("IT",),
    model=skm.LinearSVC,
    folds_n=20,
    dec_ref=0,
    gen_ref=0,
    min_trials=100,
    feat_fields=("cat_proj", "anticat_proj"),
    day_key="day",
    bhv_feats=None,
    bhv_days=None,
    combine_days=False,
    **kwargs,
):
    days = data[day_key]
    pops, feats, orders = slaux.get_fixation_stim_responses(
        data,
        tbeg=t_start,
        twid=binsize,
        regions=regions,
    )
    if combine_days:
        pops, feats, orders, days, bhv_days, bhv_feats = _combine_days_fixation(
            pops, feats, orders, days, bhv_days=bhv_days, bhv_feats=bhv_feats
        )
    test_feats_all = []
    gen_feats_all = []
    gen_proj_all = []
    test_proj_all = []
    full_outs = []
    bhv_feats_keep = []
    if gen_field is not None:
        g1_masks = feats[gen_field] > gen_ref
        g2_masks = feats[gen_field] <= gen_ref
    elif gen_func is not None:
        g1_masks = gen_func(feats)
        g2_masks = g1_masks.rs_not()
    else:
        raise IOError("one of gen_field or gen_func must be set, both are None")
    for i, pop in enumerate(pops):
        day_i = days[i]
        dec_feat = feats[dec_field][i] > dec_ref
        g1_mask = g1_masks[i]
        g2_mask = g2_masks[i]
        pop1 = pop[g1_mask]
        pop2 = pop[g2_mask]
        lab1 = dec_feat[g1_mask]
        lab2 = dec_feat[g2_mask]
        feats_con = np.concatenate(list(feats[f][i] for f in feat_fields), axis=1)
        feats1 = feats_con[g1_mask]
        feats2 = feats_con[g2_mask]
        if (
            pop1.shape[1] > 0
            and pop1.shape[0] > min_trials
            and pop2.shape[0] > min_trials
            and day_i in bhv_days
        ):
            if bhv_feats is not None:
                ind = np.where(bhv_days == day_i)[0][0]
                bhv_feats_keep.append(bhv_feats[ind])
            out1 = na.fold_skl_shape(
                pop1,
                lab1,
                folds_n,
                c_gen=pop2,
                l_gen=lab2,
                model=model,
                return_projection=True,
                **kwargs,
            )

            out2 = na.fold_skl_shape(
                pop2,
                lab2,
                folds_n,
                c_gen=pop1,
                l_gen=lab1,
                model=model,
                return_projection=True,
                **kwargs,
            )
            test_feats1 = feats1[out1["test_inds"]]
            test_feats2 = feats2[out2["test_inds"]]
            test_feats_all.append(np.concatenate((test_feats1, test_feats2), axis=2))
            gen_feats_all.append(np.concatenate((feats2, feats1)))
            gen_proj_all.append(
                np.concatenate((out1["projection_gen"], out2["projection_gen"]), axis=1)
            )
            test_proj_all.append(
                np.concatenate(
                    (out1["projection"], out2["projection"]),
                    axis=2,
                )
            )
            full_outs.append((out1, out2))

    out_dict = {
        "feats_gen": gen_feats_all,
        "proj_gen": gen_proj_all,
        "feats_test": test_feats_all,
        "proj_test": test_proj_all,
        "full_out": full_outs,
        "bhv_feats": bhv_feats_keep,
    }
    return out_dict


def similarity_pattern(
    pop,
    feats,
    targ,
    choice=None,
    balance_choice=False,
    n_folds=10,
    test_frac=0.1,
    proto_func=slaux.box_mask_array,
):
    proto_mask = proto_func(feats)
    non_proto_mask = np.logical_not(proto_mask)
    cv = na.BalancedCV(n_folds, test_size=test_frac)
    pop_proto = pop[proto_mask]
    pop_non = pop[non_proto_mask]
    targ_proto = targ[proto_mask]
    feats_proto = feats[proto_mask]
    feats_non = feats[non_proto_mask]
    u_targs = np.unique(targ)
    mu_proto = np.zeros((n_folds, len(u_targs), pop.shape[1]))
    if choice is not None and balance_choice:
        targ_proto_split = np.stack((targ_proto, choice[proto_mask]), axis=1)
    else:
        targ_proto_split = targ_proto
    te_sims_all = []
    te_feats_all = []
    non_sims = np.zeros((n_folds, len(u_targs), len(feats_non)))
    for i, (tr_inds, te_inds) in enumerate(cv.split(feats_proto, targ_proto_split)):
        pop_tr = pop_proto[tr_inds]
        targ_tr = targ_proto[tr_inds]
        for j, ut in enumerate(u_targs):
            mu_proto[i, j] = np.mean(pop_tr[ut == targ_tr])
        # te_sim = skmp.euclidean_distances(mu_proto[i], pop_proto[te_inds])
        te_sim = mu_proto[i] @ pop_proto[te_inds].T
        te_feats = feats_proto[te_inds]
        te_sims_all.append(te_sim)
        te_feats_all.append(te_feats)
        non_sims[i] = mu_proto[i] @ pop_non.T
        # non_sims[i] = skmp.euclidean_distances(mu_proto[i], pop_non)
    te_sims = np.stack(te_sims_all, axis=0)
    te_feats = np.stack(te_feats_all, axis=0)
    return te_sims, te_feats, non_sims, feats_non


def get_presentation_pops(
    data,
    *feats,
    t_start=0,
    t_end=0,
    binsize=500,
    binstep=500,
    regions=("IT",),
    min_trials=100,
    skl_axes=False,
    time_zero_field="stim_on",
    session_keys=(
        "day",
        "date",
    ),
):
    pops, xs = data.get_neural_activity(
        binsize,
        t_start,
        t_end,
        binstep,
        time_zero_field=time_zero_field,
        regions=regions,
        skl_axes=skl_axes,
    )
    session_info = data[list(session_keys)]
    feats = data[list(feats)]
    if skl_axes:
        n_neur = 0
        n_trls = 2
    else:
        n_neur = 1
        n_trls = 0
    out_pops = []
    out_feats = []
    out_session = []
    for i, pop in enumerate(pops):
        feats_i = feats[i]
        if pop.shape[n_neur] > 0 and pop.shape[n_trls] > min_trials:
            assert pop.shape[-1] == 1
            out_pops.append(pop[..., 0])
            out_feats.append(feats_i)
            out_session.append(session_info.iloc[i])
    return out_pops, out_feats, out_session


def generalize_projection_pattern(
    data,
    dec_field,
    gen_field=None,
    gen_func=None,
    t_start=0,
    t_end=0,
    binsize=500,
    binstep=500,
    regions=("IT",),
    model=skm.LinearSVC,
    time_zero_field="stim_on",
    uniform_resample=False,
    uniform_kwargs=None,
    balance_field=None,
    folds_n=20,
    dec_ref=0,
    gen_ref=0,
    min_trials=100,
    keep_feats=("chosen_cat", "cat_proj", "anticat_proj"),
    **kwargs,
):
    if uniform_resample:
        if uniform_kwargs is None:
            uniform_kwargs = {}
        labels, passing = _uniform_labels(data, **uniform_kwargs)
        data = data.session_mask(passing)
        labels = list(label for i, label in enumerate(labels) if passing[i])
    pops, xs = data.get_neural_activity(
        binsize,
        t_start,
        t_end,
        binstep,
        time_zero_field=time_zero_field,
        regions=regions,
        skl_axes=True,
    )

    dec_label = data[dec_field] > dec_ref

    if gen_func is not None:
        gen_cond = gen_func(data)
    elif gen_field is not None:
        gen_cond = data[gen_field] > gen_ref
    else:
        raise IOError("one of gen_cond or gen_func must be set, both are None")
    if balance_field is not None:
        balance_vars = data[balance_field]
    else:
        balance_vars = (None,) * len(pops)
    feats = data[list(keep_feats)]
    gen_proj_all = []
    test_proj_all = []
    test_feats_all = []
    gen_feats_all = []
    full_outs = []
    for i, pop in enumerate(pops):
        mask1 = gen_cond[i]
        labels = dec_label[i].to_numpy()
        feats_i = feats[i].to_numpy()

        pop1 = np.squeeze(pop[..., mask1, :], axis=1)
        lab1 = labels[mask1]
        feats1 = feats_i[mask1]

        mask2 = np.logical_not(gen_cond[i])
        pop2 = np.squeeze(pop[..., mask2, :], axis=1)
        lab2 = labels[mask2]
        feats2 = feats_i[mask2]

        balance_i = balance_vars[i]
        if balance_i is not None:
            balance1_i = np.stack((balance_i[mask1], lab1), axis=1)
            balance2_i = np.stack((balance_i[mask2], lab2), axis=1)
        else:
            balance1_i = None
            balance2_i = None
        if (
            pop1.shape[0] > 0
            and pop1.shape[1] > min_trials
            and pop2.shape[1] > min_trials
        ):
            out1 = na.fold_skl_flat(
                pop1,
                lab1,
                folds_n,
                c_gen=pop2,
                l_gen=lab2,
                model=model,
                return_projection=True,
                rel_flat=balance1_i,
                balance_rel_fields=balance1_i is not None,
                **kwargs,
            )

            out2 = na.fold_skl_flat(
                pop2,
                lab2,
                folds_n,
                c_gen=pop1,
                l_gen=lab1,
                model=model,
                return_projection=True,
                rel_flat=balance2_i,
                balance_rel_fields=balance2_i is not None,
                **kwargs,
            )
            test_feats1 = feats1[out1["test_inds"]]
            test_feats2 = feats2[out2["test_inds"]]
            test_feats_all.append(np.concatenate((test_feats1, test_feats2), axis=2))
            gen_feats_all.append(np.concatenate((feats2, feats1)))
            gen_proj_all.append(
                np.concatenate((out1["projection_gen"], out2["projection_gen"]), axis=1)
            )
            test_proj_all.append(
                np.concatenate(
                    (out1["projection"], out2["projection"]),
                    axis=2,
                )
            )
            full_outs.append((out1, out2))

    out_dict = {
        "feats_gen": gen_feats_all,
        "proj_gen": gen_proj_all,
        "feats_test": test_feats_all,
        "proj_test": test_proj_all,
        "xs": xs,
        "full_out": full_outs,
    }
    return out_dict


def error_projection_pattern(
    data,
    dec_field,
    t_start=0,
    t_end=0,
    binsize=500,
    binstep=500,
    regions=("IT",),
    model=skm.LinearSVC,
    time_zero_field="stim_on",
    uniform_resample=False,
    uniform_kwargs=None,
    balance_field="chosen_cat",
    folds_n=100,
    dec_ref=0,
    min_trials=100,
    keep_feats=("chosen_cat", "cat_proj", "anticat_proj"),
    **kwargs,
):
    if uniform_resample:
        if uniform_kwargs is None:
            uniform_kwargs = {}
        labels, passing = _uniform_labels(data, **uniform_kwargs)
        data = data.session_mask(passing)
        labels = list(label for i, label in enumerate(labels) if passing[i])
    pops, xs = data.get_neural_activity(
        binsize,
        t_start,
        t_end,
        binstep,
        time_zero_field=time_zero_field,
        regions=regions,
        skl_axes=True,
    )

    dec_label = data[dec_field] > dec_ref

    if balance_field is not None:
        balance_vars = data[balance_field]
    else:
        balance_vars = (None,) * len(pops)
    feats = data[list(keep_feats)]
    test_proj_all = []
    test_feats_all = []
    full_outs = []
    for i, pop in enumerate(pops):
        labels = dec_label[i].to_numpy()
        feats_i = feats[i].to_numpy()

        pop = np.squeeze(pop, axis=1)
        lab = labels

        balance_i = balance_vars[i]
        if balance_i is not None:
            balance_full_i = np.stack((balance_i, lab), axis=1)
        else:
            balance_full_i = None
        if pop.shape[0] > 0 and pop.shape[1] > min_trials:
            out = na.fold_skl_flat(
                pop,
                lab,
                folds_n,
                model=model,
                return_projection=True,
                rel_flat=balance_full_i,
                balance_rel_fields=balance_full_i is not None,
                **kwargs,
            )

            test_feats = feats_i[out["test_inds"]]
            test_feats_all.append(test_feats)
            test_proj_all.append(out["projection"])
            full_outs.append(out)

    out_dict = {
        "feats_test": test_feats_all,
        "proj_test": test_proj_all,
        "xs": xs,
        "full_out": full_outs,
    }
    return out_dict


def joint_variable_decoder(
    pops,
    feature_vars,
    model=skm.LinearSVC,
    shape_labels=None,
    keep_trial_info=None,
    n_folds=20,
    min_trials=100,
    indiv_zscore=True,
    rel_flat_all=None,
    **kwargs,
):
    if shape_labels is None:
        shape_labels = (None,) * len(pops)
    if keep_trial_info is None:
        keep_trial_info = (None,) * len(pops)
    n_neurs = np.max(list(p.shape[0] for p in pops))
    n_ts = pops[0].shape[-1]
    n_feats = feature_vars[0].shape[1]
    dv_shape = (n_folds, n_neurs, n_feats, n_ts)
    dec_vecs = []
    proc_pops = []
    test_inds = []
    feats = []
    session_info = []
    ests_all = []
    trial_info = []
    for i, pop_i in enumerate(pops):
        fv_i = feature_vars[i]
        if rel_flat_all is not None:
            kwargs["rel_flat"] = rel_flat_all[i]
        if len(fv_i) >= min_trials and pop_i.shape[0] > 0:
            labels_i = []
            for j in range(fv_i.shape[1]):
                fv_ij = fv_i[:, j]
                if len(np.unique(fv_ij)) > 2:
                    labels_i.append(fv_ij > 0)
                else:
                    labels_i.append(fv_ij > np.mean(fv_ij))
            labels_i = np.stack(labels_i, axis=1)
            if indiv_zscore:
                (pop_i,) = na.zscore_tc(pop_i)
            pop_i = np.squeeze(pop_i, axis=1)
            proc_pops.append(pop_i)
            feats.append(fv_i)
            out = na.fold_skl_flat(
                pop_i,
                labels_i,
                n_folds,
                mean=False,
                model=model,
                pre_pca=None,
                norm=False,
                **kwargs,
            )
            ests = out["estimators"]
            ests_all.append(ests)
            test_inds.append(out["test_inds"])
            n_neur_i = pop_i.shape[0]
            dv_i = np.zeros(dv_shape)
            for k, l_ in u.make_array_ind_iterator(ests.shape):
                coefs = np.squeeze(
                    np.stack(
                        list(est.coef_ for est in ests[k, l_][-1].estimators_), axis=0
                    )
                )
                dv_i[k, :n_neur_i, :, l_] = coefs.T
            dec_vecs.append(dv_i)
            session_info.append(shape_labels[i])
            trial_info.append(keep_trial_info[i])
    out_dict = {
        "dvs": dec_vecs,
        "test_inds": test_inds,
        "pops": proc_pops,
        "feats": feats,
        "session_info": session_info,
        "trial_info": trial_info,
        "estimators": ests_all,
    }
    return out_dict


def combined_pregen_decoder(data_dict, pre_shapes, gen_shape, **kwargs):
    out_dict = {}
    out_pre = combined_choice_decoder(data_dict, shapes=pre_shapes, **kwargs)
    out_gen = joint_variable_shape_sequence(data_dict, shapes=(gen_shape,), **kwargs)
    out_dict.update(out_gen)
    out_dict.update(out_pre)
    return out_dict


def resample_uniform_performance(
    data, cho_field="chosen_cat", targ_field="stim_sample_MAIN", n_samps=1000
):
    corr = data[cho_field] == data[targ_field]
    perf = np.zeros((len(corr), n_samps))
    masks = slaux.sample_uniform_mask(data, n_asamps=n_samps)
    for j, corr_j in enumerate(corr):
        for i in range(n_samps):
            perf[j, i] = np.nanmean(corr[masks[j][:, i]])
    perf = perf[np.all(~np.isnan(perf), axis=1)]
    return perf


def cross_session_generalization(
    *dm_pairs,
    winsize=500,
    tbeg=-500,
    tend=500,
    stepsize=50,
    shape_field="shape",
    date_field="date",
    region="IT",
    time_zero_field="stim_on",
    indiv_zscore=True,
    n_folds=20,
    **kwargs,
):
    shapes, dates, xs, (pops1, pops2) = _format_cross_session_info(
        *dm_pairs,
        winsize=winsize,
        tbeg=tbeg,
        tend=tend,
        stepsize=stepsize,
        region=region,
        time_zero_field=time_zero_field,
    )
    out = _generalize_cross_session_decoder(
        shapes,
        dates,
        pops1,
        pops2,
        indiv_zscore=indiv_zscore,
        n_folds=n_folds,
        **kwargs,
    )
    shapes, dates, gen_arr, gen_var = out
    return shapes, dates, xs, gen_arr, gen_var


def cross_data_generalization(
    data1,
    masks1,
    data2,
    masks2,
    *args,
    ind_pairs=None,
    params=None,
    max_iter=1000,
    model=skm.LinearSVC,
    pre_pca=0.99,
    shuffle=False,
    n_folds=100,
    time_zero_field="stim_on",
    region="IT",
    flip=True,
    indiv_zscore=True,
    **kwargs,
):
    if ind_pairs is None:
        ind_pairs = list(zip(range(len(masks1)), range(len(masks2))))
    if params is None:
        params = {
            "class_weight": "balanced",
            "max_iter": max_iter,
        }

    out1 = data1.get_dec_pops(
        *args,
        *masks1,
        tzfs=(time_zero_field,) * 2,
        shuffle_trials=False,
        regions=(region,),
    )
    xs, (pops1_m1, pops1_m2) = out1
    pops1_m1 = list(filter(lambda x: x.shape[0] > 0, pops1_m1))
    pops1_m2 = list(filter(lambda x: x.shape[0] > 0, pops1_m2))

    out2 = data2.get_dec_pops(
        *args,
        *masks2,
        tzfs=(time_zero_field,) * 2,
        shuffle_trials=False,
        regions=(region,),
    )
    xs, (pops2_m1, pops2_m2) = out2
    pops2_m1 = list(filter(lambda x: x.shape[0] > 0, pops2_m1))
    pops2_m2 = list(filter(lambda x: x.shape[0] > 0, pops2_m2))

    n_decs = len(ind_pairs)
    outs = np.zeros((n_decs, n_folds, len(xs)))
    outs_gen = np.zeros_like(outs)
    if flip:
        outs_flip = np.zeros_like(outs)
        outs_gen_flip = np.zeros_like(outs)

    for i, (ind1, ind2) in enumerate(ind_pairs):
        p11 = pops1_m1[ind1]
        p12 = pops1_m2[ind1]
        p21 = pops2_m1[ind2]
        p22 = pops2_m2[ind2]
        if indiv_zscore:
            p11, p12 = na.zscore_tc(p11, p12)
            p21, p22 = na.zscore_tc(p21, p22)
            kwargs["norm"] = False

        out = na.fold_skl(
            p11,
            p12,
            n_folds,
            model=model,
            params=params,
            pre_pca=pre_pca,
            shuffle=shuffle,
            gen_c1=p21,
            gen_c2=p22,
            mean=False,
            **kwargs,
        )
        outs[i] = out["score"]
        outs_gen[i] = out["score_gen"]
        if flip:
            out = na.fold_skl(
                p21,
                p22,
                n_folds,
                model=model,
                params=params,
                pre_pca=pre_pca,
                shuffle=shuffle,
                gen_c1=p11,
                gen_c2=p12,
                mean=False,
                **kwargs,
            )
            outs_flip[i] = out["score"]
            outs_gen_flip[i] = out["score_gen"]

    out_dict = {
        "dec": outs,
        "xs": xs,
        "gen": outs_gen,
    }
    if flip:
        out_dict["dec_flip"] = outs_flip
        out_dict["gen_flip"] = outs_gen_flip
    return out_dict


def decode_category_session_gen(
    data,
    *args,
    cat_field="stim_sample_MAIN",
    time_zero_field="stim_on",
    order_key="day",
    params=None,
    max_iter=1000,
    model=skm.LinearSVC,
    pre_pca=0.99,
    shuffle=False,
    session_offset=-1,
    n_folds=100,
    region="IT",
    **kwargs,
):
    targs = data[cat_field]
    t1_mask = targs == 1
    t2_mask = targs == 2

    days = np.array(data[order_key])
    inds = np.argsort(days)
    days = days[inds]
    out = data.get_dec_pops(
        *args,
        t1_mask,
        t2_mask,
        tzfs=(time_zero_field,) * 2,
        shuffle_trials=False,
        regions=(region,),
    )
    xs, (pops_t1, pops_t2) = out
    pops_t1 = np.array(pops_t1, dtype=object)[inds]
    pops_t2 = np.array(pops_t2, dtype=object)[inds]
    pops_t1 = list(filter(lambda x: x.shape[0] > 0, pops_t1))
    pops_t2 = list(filter(lambda x: x.shape[0] > 0, pops_t2))

    if params is None:
        params = {"class_weight": "balanced", "max_iter": max_iter, "dual": "auto"}

    if session_offset < 0:
        start = -session_offset
        end = len(pops_t1)
        i_use = session_offset
    else:
        start = 0
        end = len(pops_t1) - session_offset
        i_use = session_offset
    outs = np.zeros((len(pops_t1), n_folds, len(xs)))
    outs_gen = np.zeros_like(outs)

    day_out = np.zeros((len(pops_t1), 2))
    for i in range(start, end):
        pop_i_t1 = pops_t1[i]
        pop_i_t2 = pops_t2[i]
        day_out[i, 0] = days[i]
        day_out[i, 1] = days[i + i_use]
        out = na.fold_skl(
            pop_i_t1,
            pop_i_t2,
            n_folds,
            model=model,
            params=params,
            pre_pca=pre_pca,
            shuffle=shuffle,
            gen_c1=pops_t1[i + i_use],
            gen_c2=pops_t2[i + i_use],
            **kwargs,
        )
        outs[i] = out[0]
        outs_gen[i] = out[1]
    outs = outs[start:end]
    outs_gen = outs_gen[start:end]
    day_out = day_out[start:end]
    return outs, xs, outs_gen, day_out


def decode_category(
    data,
    *args,
    cat_field="stim_sample_MAIN",
    time_zero_field="stim_on",
    cho_field="chosen_cat",
    targ_field="stim_sample_MAIN",
    day_field="day",
    uniform_resample=False,
    estimate_performance=False,
    **kwargs,
):
    if uniform_resample:
        data = uniform_sample_mask(data)
    targs = data[cat_field]
    t1_mask = targs == 1
    t2_mask = targs == 2
    out = data.decode_masks(
        t1_mask,
        t2_mask,
        *args,
        time_zero_field=time_zero_field,
        decode_m1=t1_mask,
        decode_m2=t2_mask,
        decode_tzf=time_zero_field,
        **kwargs,
    )
    if estimate_performance:
        corr = data[cho_field] == data[targ_field]
        days = data[day_field]
        out = out + (corr, days)
    return _make_out_dict(data, out)


def decode_feature_values(
    data,
    *args,
    feat_field="cat_proj",
    time_zero_field="stim_on",
    uniform_resample=False,
    **kwargs,
):
    if uniform_resample:
        data = uniform_sample_mask(data)
    targ_feat = data[feat_field]
    mask = targ_feat.rs_isnan().rs_not()
    data_use = data.mask(mask)

    out = data_use.regress_target_field(
        feat_field,
        *args,
        time_zero_field=time_zero_field,
        ret_pred_targ=True,
        **kwargs,
    )
    return _make_out_dict(data, out)


def compute_var_ratio(data, func=np.std):
    n_ds = len(data.data)
    cat_var = np.zeros(n_ds)
    acat_var = np.zeros_like(cat_var)
    for i in range(n_ds):
        cat_var[i] = func(data["cat_proj"][i])
        acat_var[i] = func(data["anticat_proj"][i])
    return acat_var / cat_var


def decode_cat_feature(*args, **kwargs):
    return decode_feature_values(*args, **kwargs, feat_field="cat_proj")


def decode_anticat_feature(*args, **kwargs):
    return decode_feature_values(*args, **kwargs, feat_field="anticat_proj")


def generalize_feature_values(
    data,
    *args,
    feat_field="cat_proj",
    gen_field="anticat_proj",
    gap=0,
    time_zero_field="stim_on",
    uniform_resample=False,
    **kwargs,
):
    if uniform_resample:
        data = uniform_sample_mask(data)
    targ_feat = data[feat_field]
    mask = targ_feat.rs_isnan().rs_not()
    data_use = data.mask(mask)

    dec_mask = data_use[gen_field] > gap / 2
    gen_mask = data_use[gen_field] < -gap / 2
    out = data_use.regress_target_field(
        feat_field,
        *args,
        time_zero_field=time_zero_field,
        train_mask=dec_mask,
        gen_mask=gen_mask,
        ret_pred_targ=True,
        **kwargs,
    )
    return _make_out_dict(data, out)


def generalize_cat_feature(*args, **kwargs):
    return generalize_feature_values(*args, **kwargs)


def generalize_anticat_feature(*args, **kwargs):
    return generalize_feature_values(
        *args, **kwargs, gen_field="cat_proj", feat_field="anticat_proj"
    )


def decode_xor(
    data,
    *args,
    n_chambers=2,
    cat_field="stim_sample_MAIN",
    cat_bound_field="cat_def_MAIN",
    time_zero_field="stim_on",
    eps=1e-10,
    uniform_resample=False,
    estimate_performance=False,
    cho_field="chosen_cat",
    targ_field="stim_sample_MAIN",
    day_field="day",
    **kwargs,
):
    if uniform_resample:
        data = uniform_sample_mask(data)

    coherence = np.zeros(data.n_sessions)
    boundary_vecs = np.zeros((data.n_sessions, 2, 2))
    if cat_field == "stim_sample_MAIN":
        trls = data["targ_cho"] == data["targ_cor"]
        for i, cba in enumerate(data[cat_bound_field]):
            cba_u, counts = np.unique(cba, return_counts=True)
            cba = cba_u[np.argmax(counts)]
            boundary_vecs[i, 0] = (np.cos(np.radians(cba)), -np.sin(np.radians(cba)))
            boundary_vecs[i, 1] = (np.cos(np.radians(cba)), np.sin(np.radians(cba)))
            coherence[i] = np.nanmean(trls[i])
    else:
        for i in range(data.n_sessions):
            db = estimate_decision_boundary(data, ind=i)
            eb = db["boundary"]
            boundary_vecs[i, 0] = (eb[0][0], eb[0][1])
            boundary_vecs[i, 1] = (eb[0][0], -eb[0][1])
            coherence[i] = db["coherence"]

    stim_feats = data["stim_feature_MAIN"]
    d1_masks = []
    d2_masks = []
    for i in range(data.n_sessions):
        proj = np.squeeze(
            np.stack(stim_feats[i].to_numpy(), axis=0) / 1000 @ boundary_vecs[i].T
        )
        p_min = np.min(proj, axis=0)
        p_max = np.max(proj, axis=0)
        bins = np.linspace(p_min - eps, p_max + eps, n_chambers + 1)
        x_bins = np.digitize(proj[:, 0], bins[:, 0]) - 1
        y_bins = np.digitize(proj[:, 1], bins[:, 1]) - 1
        xb = np.unique(x_bins)
        yb = np.unique(y_bins)

        d1_i = np.zeros(len(proj), dtype=bool)
        d2_i = np.zeros_like(d1_i)
        for xb_i, yb_i in it.product(xb, yb):
            group = np.logical_and(x_bins == xb_i, y_bins == yb_i)
            if xb_i % 2 == yb_i % 2:
                d1_i = np.logical_or(d1_i, group)
            else:
                d2_i = np.logical_or(d2_i, group)
        d1_masks.append(d1_i)
        d2_masks.append(d2_i)

    out = data.decode_masks(
        d1_masks,
        d2_masks,
        *args,
        decode_m1=d1_masks,
        decode_m2=d2_masks,
        time_zero_field=time_zero_field,
        decode_tzf=time_zero_field,
        **kwargs,
    )
    if estimate_performance:
        corr = data[cho_field] == data[targ_field]
        days = data[day_field]
        out = out + (corr, days)

    return _make_out_dict(data, out)


def generalize_category(
    data,
    *args,
    cat_field="stim_sample_MAIN",
    cat_bound_field="cat_def_MAIN",
    time_zero_field="stim_on",
    print_proportion=False,
    ortho_category_decode=False,
    ortho_category_generalize=False,
    uniform_resample=False,
    estimate_performance=False,
    cho_field="chosen_cat",
    targ_field="stim_sample_MAIN",
    day_field="day",
    thr=0,
    **kwargs,
):
    if uniform_resample:
        data = uniform_sample_mask(data)

    targs = data[cat_field]
    t1_masks = targs == 1
    t2_masks = targs == 2
    coherence = np.zeros(data.n_sessions)
    boundary_vecs = np.zeros((data.n_sessions, 2))
    if cat_field == "stim_sample_MAIN":
        trls = data["targ_cho"] == data["targ_cor"]
        for i, cba in enumerate(data[cat_bound_field]):
            cba_u, counts = np.unique(cba, return_counts=True)
            cba = cba_u[np.argmax(counts)]
            boundary_vecs[i] = (np.cos(np.radians(cba)), np.sin(np.radians(cba)))
            coherence[i] = np.nanmean(trls[i])
    else:
        for i in range(data.n_sessions):
            db = estimate_decision_boundary(data, ind=i)
            eb = db["boundary"]
            boundary_vecs[i] = (eb[0][0], -eb[0][1])
            coherence[i] = db["coherence"]

    stim_feats = data["stim_feature_MAIN"]
    d1_masks = []
    d2_masks = []
    g1_masks = []
    g2_masks = []
    for i in range(data.n_sessions):
        proj = np.squeeze(
            np.stack(stim_feats[i].to_numpy(), axis=0)
            / 1000
            @ boundary_vecs[i : i + 1].T
        )
        pos_proj = proj > thr
        neg_proj = proj <= thr
        if print_proportion:
            print(np.mean(pos_proj))
        if ortho_category_decode:
            d1_masks.append(pos_proj)
            d2_masks.append(neg_proj)
            g1_masks.append(pos_proj)
            g2_masks.append(neg_proj)
        elif ortho_category_generalize:
            d1_masks.append(np.logical_and(t1_masks[i], pos_proj))
            d2_masks.append(np.logical_and(t1_masks[i], neg_proj))
            g1_masks.append(np.logical_and(t2_masks[i], pos_proj))
            g2_masks.append(np.logical_and(t2_masks[i], neg_proj))
        else:
            d1_masks.append(np.logical_and(t1_masks[i], pos_proj))
            d2_masks.append(np.logical_and(t2_masks[i], pos_proj))
            g1_masks.append(np.logical_and(t1_masks[i], neg_proj))
            g2_masks.append(np.logical_and(t2_masks[i], neg_proj))

    out = data.decode_masks(
        d1_masks,
        d2_masks,
        *args,
        decode_m1=g1_masks,
        decode_m2=g2_masks,
        time_zero_field=time_zero_field,
        decode_tzf=time_zero_field,
        **kwargs,
    )
    if estimate_performance:
        corr = data[cho_field] == data[targ_field]
        days = data[day_field]
        out = out + (corr, days)

    return _make_out_dict(data, out)


def decode_ortho_category(*args, **kwargs):
    return generalize_category(*args, **kwargs, ortho_category_decode=True)


def generalize_ortho_category(*args, **kwargs):
    return generalize_category(*args, **kwargs, ortho_category_generalize=True)
