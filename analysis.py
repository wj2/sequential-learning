import itertools as it
import numpy as np
import sklearn.svm as skm
import sklearn.gaussian_process as skgp
import pandas as pd

import general.utility as u
import general.neural_analysis as na

import sequential_learning.auxiliary as slaux


def uniform_sample_mask(data, **kwargs):
    mask = slaux.sample_uniform_mask(data, **kwargs)
    data = data.mask(mask)
    session_mask = list(np.sum(x) > 0 for x in mask)
    data = data.session_mask(session_mask)
    return data


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
    **kwargs,
):
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
    for i, p1_i in enumerate(pop1):
        p2_i = pop2[i]
        if p1_i.shape[0] > 0:
            mu1 = np.squeeze(np.mean(p1_i, axis=2))
            sig1 = np.squeeze(np.std(p1_i, axis=2))
            mu2 = np.squeeze(np.mean(p2_i, axis=2))
            sig2 = np.squeeze(np.std(p2_i, axis=2))
            dprimes[i] = np.nanmean(np.abs(mu1 - mu2) / np.sqrt(sig1 * sig2), axis=0)
        else:
            dprimes[i] = np.nan
    return xs, dprimes, days


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
    for i, p1_i in enumerate(pops1):
        p2_i = pops2[i]
        out = na.fold_skl(p1_i, p2_i, n_folds, mean=False, model=model, **kwargs)
        out_gen[i, i] = out["score"]
        for j, p1_j in enumerate(pops1):
            p2_j = pops2[j]
            if i != j:
                out_gen[i, j] = na.apply_estimators_discrete(
                    out["estimators"],
                    p1_j,
                    p2_j,
                )
    return shapes, dates, out_gen


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
    shapes, dates, gen_arr = out
    return shapes, dates, xs, gen_arr


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
    uniform_resample=False,
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
    return _make_out_dict(data, out)


def decode_ortho_category(*args, **kwargs):
    return generalize_category(*args, **kwargs, ortho_category_decode=True)


def generalize_ortho_category(*args, **kwargs):
    return generalize_category(*args, **kwargs, ortho_category_generalize=True)
