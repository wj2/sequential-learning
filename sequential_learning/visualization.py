import matplotlib.pyplot as plt
import numpy as np
import itertools as it
import scipy.signal as ssig
import scipy.stats as sts

import sklearn.decomposition as skd
import sklearn.linear_model as sklm
import sklearn.preprocessing as skp
import sklearn.svm as skm

import general.plotting as gpl
import general.utility as u
import sequential_learning.analysis as sla
import sequential_learning.auxiliary as slaux


@gpl.ax_adder()
def plot_shape_scatter(
    pops,
    k1,
    k2,
    comb="{}post{}",
    first_sess=0,
    last_sess=-1,
    ax=None,
    norm_time=-250,
    quant_time=250,
    color1="r",
    color2="g",
    func=np.mean,
):
    k1k2_comb = comb.format(k1, k2)

    p1, xs = pops[k1]
    p2, xs = pops[k2]
    p12, xs = pops[k1k2_comb]
    norm_ind = np.argmin((xs - norm_time) ** 2)
    quant_ind = np.argmin((xs - quant_time) ** 2)

    pops = []
    for p in (p1, p2, p12):
        p_first = np.swapaxes(np.squeeze(p[first_sess]), 0, 1)
        s_first = skp.StandardScaler().fit(p_first[..., norm_ind])
        pop_first = func(s_first.transform(p_first[..., quant_ind]), axis=0)
        p_last = np.swapaxes(np.squeeze(p[last_sess]), 0, 1)
        s_last = skp.StandardScaler().fit(p_last[..., norm_ind])
        pop_last = func(s_last.transform(p_last[..., quant_ind]), axis=0)
        pops.append((np.expand_dims(pop_first, -1), np.expand_dims(pop_last, -1)))
    (_, pop_k1_early), (pop_k2_early, pop_k2_late), (pop_k1_late, _) = pops

    k1_el = np.concatenate((pop_k1_early, pop_k1_late), axis=1)
    k2_el = np.concatenate((pop_k2_early, pop_k2_late), axis=1)
    ax.plot(k1_el.T, k2_el.T, color=(0.9, 0.9, 0.9))
    ax.plot(k1_el[:, 0], k2_el[:, 0], "o", color=color1, label="early")
    ax.plot(k1_el[:, 1], k2_el[:, 1], "o", color=color2, label="late")
    gpl.clean_plot(ax, 0)
    ax.set_aspect("equal")
    ax.set_ylabel("activity for shape {}".format(k2))
    ax.set_xlabel("activity for shape {}".format(k1))
    ax.legend(frameon=False)


@gpl.ax_adder()
def plot_session_change_scatter(
    xs,
    metric,
    days,
    data,
    cm="Blues",
    var_thr=0.6,
    ax=None,
    t_cent=150,
    mask_var=True,
):
    cm = plt.get_cmap(cm)
    if mask_var:
        mask = sla.compute_var_ratio(data) > var_thr
        metric = metric[mask]
        days = days[mask]
    colors = cm(np.linspace(0.2, 1, len(metric)))
    t_ind = np.argmin(np.abs(xs - t_cent))

    for i, metric_i in enumerate(metric):
        ax.plot(days[i], metric_i[t_ind], "o", color=colors[i])
        gpl.clean_plot(ax, 0)


def plot_session_change(
    xs,
    metric,
    days,
    data,
    cm="Blues",
    var_thr=0.6,
    axs=None,
    fwid=2,
    t_cent=150,
    mask_var=True,
):
    cm = plt.get_cmap(cm)
    if mask_var:
        mask = sla.compute_var_ratio(data) > var_thr
        metric = metric[mask]
        days = days[mask]
    colors = cm(np.linspace(0.2, 1, len(metric)))

    if axs is None:
        f, axs = plt.subplots(1, 2, figsize=(fwid * 2, fwid))
    ax1, ax2 = axs
    t_ind = np.argmin(np.abs(xs - t_cent))

    for i, metric_i in enumerate(metric):
        gpl.plot_trace_werr(xs, metric_i, ax=ax1, color=colors[i])
        ax2.plot(days[i], metric_i[t_ind], "o", color=colors[i])
        gpl.clean_plot(ax2, 0)


@gpl.ax_adder()
def plot_related_unrelated_performance(
    data_seq,
    ax=None,
    cat_field="cat_def_MAIN",
    related_color="r",
    unrelated_color="g",
    **kwargs,
):
    cats = []
    for k, data in data_seq.items():
        cat_bound = data[cat_field][0].iloc[0]
        print(cat_bound, cats)
        if cat_bound in cats:
            color = related_color
        else:
            color = unrelated_color
        cats.append(cat_bound)
        plot_session_average(data, ax=ax, color=color, **kwargs)


@gpl.ax_adder()
def plot_session_average(
    data,
    ax=None,
    fwid=1,
    cho_field="chosen_cat",
    targ_field="stim_sample_MAIN",
    day_field="day",
    n_boots=500,
    uniform_resample=True,
    **kwargs,
):
    days = data[day_field]
    days, inds = np.unique(days, return_index=True)
    choices = data[cho_field]
    targets = data[targ_field]
    corr = np.zeros((n_boots, len(days)))
    for i, ind in enumerate(inds):
        if uniform_resample:
            corr[:, i] = sla.resample_uniform_performance(data, ind, n_samps=n_boots)
        else:
            perf = choices[ind].to_numpy() == targets[ind].to_numpy()
            corr[:, i] = u.bootstrap_list(perf, np.nanmean, n_boots)
    gpl.plot_trace_werr(days, corr, ax=ax, conf95=True, **kwargs)
    gpl.add_hlines(0.5, ax)


def compute_basis(basis_vecs):
    bv_all = np.concatenate(basis_vecs, axis=-1)
    basis, _ = np.linalg.qr(bv_all)
    return basis


def plot_all_cross_projections(*dvs, cmaps=None, color_bounds=(0.2, 0.9), **kwargs):
    n_dvs = len(dvs)
    if cmaps is None:
        cmaps = ("Blues", "Greens", "Reds")
    cmaps = list(plt.get_cmap(x) for x in cmaps)
    axs = kwargs.pop("axs", None)
    for i, dv in enumerate(dvs):
        ref_inds_base = (None,) * i
        colors = cmaps[i](np.linspace(*color_bounds, len(dv)))
        for j, dv_j in enumerate(dv):
            ref_inds_use = ref_inds_base + (j,) + (None,) * (n_dvs - i - 1)
            axs = plot_cross_projection(
                *dvs, ref_inds=ref_inds_use, axs=axs, color=colors[j], **kwargs
            )


def plot_cross_projection(*dvs, ref_inds=None, axs=None, fwid=3, t_ind=0, **kwargs):
    n = len(dvs)
    if axs is None:
        f, axs = plt.subplots(n, n, figsize=(n * fwid, n * fwid), sharey=True)
    if ref_inds is None:
        ref_inds = (-1,) * len(dvs)
    sims_all = {}
    for i, dv_i in enumerate(dvs):
        if ref_inds[i] is not None:
            dv_ref = dv_i[ref_inds[i]]
            basis_ref = compute_basis((dv_ref[..., t_ind],))
            for j, dv_j in enumerate(dvs):
                sim = np.zeros((basis_ref.shape[0], len(dv_j)))
                for k, dv_jk in enumerate(dv_j):
                    basis_comp = compute_basis((dv_jk[..., t_ind],))
                    sim[:, k] = (
                        np.sum(
                            np.array(
                                list(
                                    x.T @ basis_comp[z] for z, x in enumerate(basis_ref)
                                )
                            )
                            ** 2,
                            axis=(1, 2),
                        )
                        / 2
                    )
                sims_all[(i, j)] = sim
    for ind, sim in sims_all.items():
        if ind[0] == ind[1]:
            sim[:, ref_inds[ind[0]]] = np.nan
        gpl.plot_trace_werr(
            np.arange(sim.shape[1]), sim, ax=axs[ind], confstd=True, **kwargs
        )
        gpl.clean_plot(axs[ind], ind[1])
        if ind[0] == 0:
            axs[ind].set_ylabel("alignment index")
        if ind[1] == n - 1:
            axs[ind].set_xlabel("session number")
    return axs


def _boxcar(ref_feat, feats, radius=0.2):
    dists = np.sqrt(np.sum((np.expand_dims(ref_feat, 0) - feats) ** 2, axis=1))
    weights = dists < radius
    return weights / np.sum(weights)


def average_similar_stimuli(
    reps,
    feats,
    weight_func=_boxcar,
    **kwargs,
):
    new_reps = np.zeros_like(reps)
    for i, rep_i in enumerate(reps):
        weights = weight_func(feats[i], feats, **kwargs)
        new_reps[i] = np.sum(np.expand_dims(weights, 1) * reps, axis=0)
    return new_reps


def plot_fixation_generalization(
    feats_bhv,
    projs,
    feats,
    feat_dims_bhv=(1, 2),
    choice_dim_bhv=0,
    feat_dims=(0, 1),
    axs=None,
    fwid=2,
    average_folds=True,
    smooth_radius=0.2,
):
    if axs is None:
        f, axs = plt.subplots(
            len(projs), 2, figsize=(fwid * 2, fwid * len(projs)), squeeze=False
        )

    for i, proj_i in enumerate(projs):
        feats_bhv_i = np.reshape(feats_bhv[i], (-1, feats_bhv[i].shape[-1]))
        feats_i = np.reshape(feats[i], (-1, feats[i].shape[-1]))
        rel_feats_bhv = feats_bhv_i[..., feat_dims_bhv]
        rel_feats = feats_i[..., feat_dims]
        rel_choice = feats_bhv_i[..., choice_dim_bhv]
        if average_folds:
            ax = (0, -1)
        else:
            ax = 1
        proj_i = np.mean(proj_i, axis=ax).flatten()
        plot_gen_map_average(
            rel_feats_bhv,
            rel_choice,
            ax=axs[i, 0],
            vmin=1,
            vmax=2,
            smooth_radius=smooth_radius,
        )
        plot_gen_map_average(
            rel_feats,
            proj_i > 0,
            ax=axs[i, 1],
            vmin=0,
            vmax=1,
            smooth_radius=smooth_radius,
        )


def plot_full_generalization(
    projs,
    feats,
    feat_dims=(1, 2),
    choice_dim=0,
    axs=None,
    fwid=2,
    average_folds=False,
    smooth_radius=0.2,
):
    if axs is None:
        f, axs = plt.subplots(len(projs), 2, figsize=(fwid * 2, fwid * len(projs)))

    for i, proj_i in enumerate(projs):
        feats_i = np.reshape(feats[i], (-1, feats[i].shape[-1]))
        rel_feats = feats_i[..., feat_dims]
        rel_choice = feats_i[..., choice_dim]
        if average_folds:
            ax = (0, -1)
        else:
            ax = 1
        proj_i = np.mean(proj_i, axis=ax).flatten()
        plot_gen_map_average(
            rel_feats,
            rel_choice,
            ax=axs[i, 0],
            vmin=1,
            vmax=2,
            smooth_radius=smooth_radius,
        )
        plot_gen_map_average(
            rel_feats,
            proj_i > 0,
            ax=axs[i, 1],
            vmin=0,
            vmax=1,
            smooth_radius=smooth_radius,
        )


def make_average_map(
    x_vals,
    y_vals,
    stim,
    weights,
    weight_func=_boxcar,
    radius=0.2,
):
    xs, ys = np.meshgrid(x_vals, y_vals)
    xs_flat = xs.flatten()
    ys_flat = ys.flatten()

    pts = np.stack((xs_flat, ys_flat), axis=1)
    map_vals = np.zeros(len(pts))
    for i, pt in enumerate(pts):
        pt_i = np.sum(weights * weight_func(pt, stim, radius=radius), axis=0)
        map_vals[i] = pt_i

    map_use = map_vals.reshape((len(x_vals), len(y_vals)))
    return map_use


@gpl.ax_adder()
def plot_gen_scatter(
    feat_i,
    feat_choice,
    proj_i,
    ax=None,
    feat_dims=(1, 2),
    cmap="bwr",
    plot_choice=True,
    choice_ind=0,
    pt_size=5,
    prop_inc=2,
):
    proj_max = np.max(np.abs(proj_i))
    if plot_choice:
        ax.scatter(*feat_i.T, c=feat_choice, cmap=cmap, s=pt_size * prop_inc)
    ax.scatter(*feat_i.T, c=proj_i, cmap=cmap, vmin=-proj_max, vmax=proj_max, s=pt_size)
    ax.set_aspect("equal")
    gpl.add_hlines(0, ax)
    gpl.add_vlines(0, ax, color="k")
    gpl.clean_plot(ax, 1)
    gpl.clean_plot_bottom(ax)


@gpl.ax_adder()
def plot_gen_map(feats, proj, ax=None, cmap="bwr", model=skm.SVC, n_pts=50):
    m = model().fit(feats, proj)

    bound = np.max(np.abs(feats))
    pts = np.linspace(-bound, bound, n_pts)

    xs, ys = np.meshgrid(pts, pts)
    xs_flat = xs.reshape((-1,))
    ys_flat = ys.reshape((-1,))
    preds = m.decision_function(np.stack((xs_flat, ys_flat), axis=1))

    preds = preds.reshape((n_pts, n_pts))
    v_bound = np.max(np.abs(preds))
    gpl.pcolormesh(pts, pts, preds, cmap=cmap, vmin=-v_bound, vmax=v_bound, ax=ax)
    ax.set_aspect("equal")
    gpl.add_hlines(0, ax)
    gpl.add_vlines(0, ax, color="k")
    gpl.clean_plot(ax, 1)
    gpl.clean_plot_bottom(ax)


@gpl.ax_adder()
def plot_gen_map_average(
    feats,
    proj,
    ax=None,
    cmap="bwr",
    model=skm.SVC,
    n_pts=50,
    vmin=None,
    vmax=None,
    smooth_radius=0.2,
    plot_scatter=False,
):
    bound = np.max(np.abs(feats))
    pts = np.linspace(-bound, bound, n_pts)

    preds = make_average_map(pts, pts, feats, proj, radius=smooth_radius)
    v_bound = np.max(np.abs(preds))
    if vmin is None:
        vmin = -v_bound
    if vmax is None:
        vmax = v_bound
    gpl.pcolormesh(pts, pts, preds, cmap=cmap, vmin=vmin, vmax=vmax, ax=ax)
    if plot_scatter:
        ax.scatter(*feats.T, c=proj, s=0.1, vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_aspect("equal")
    gpl.add_hlines(0, ax)
    gpl.add_vlines(0, ax, color="k")
    gpl.clean_plot(ax, 1)
    gpl.clean_plot_bottom(ax)


def plot_shape_gen_maps(
    pops,
    feats,
    axs=None,
    fwid=2,
    n_folds=100,
    balance_choice=True,
    feat_inds=(0, 1),
    dec_ind=0,
    dec_ref=0,
    choice_ind=-1,
):
    if axs is None:
        f, axs = plt.subplots(len(pops), 2, figsize=(fwid * 2, fwid * len(pops)))
    for sess_ind, pop_i in enumerate(pops):
        pop_i = pops[sess_ind]
        feats_i = feats[sess_ind].to_numpy()

        pop_i_zs = skp.StandardScaler().fit_transform(pop_i)

        te_sims, te_feats, non_sims, non_feats = sla.similarity_pattern(
            pop_i_zs,
            feats_i,
            feats_i[:, dec_ind] > dec_ref,
            choice=feats_i[:, choice_ind],
            n_folds=n_folds,
            balance_choice=balance_choice,
        )
        sim_diff = np.diff(np.mean(non_sims, axis=0), axis=0)[0]
        if non_sims.shape[-1] > 0:
            plot_gen_map_average(
                non_feats[:, feat_inds],
                non_feats[:, choice_ind],
                vmin=1,
                vmax=2,
                ax=axs[sess_ind, 0],
            )
            plot_gen_map_average(
                non_feats[:, feat_inds],
                sim_diff,
                vmin=-1,
                vmax=1,
                ax=axs[sess_ind, 1],
            )

        # can we see an off-boundary effect on choice?
        print(te_sims.shape, non_sims.shape)


def project_features_common(
    dvs_all,
    *pairs,
    color_maps=None,
    t_ind=0,
    ax=None,
    fv_ind=0,
    ms=1,
    radius=0.15,
    vmin=-1,
    vmax=1,
):
    if color_maps is None:
        color_maps = ("magma",) * len(pairs)
    if ax is None:
        f, ax = plt.subplots(1, 1, subplot_kw={"projection": "3d"})
    else:
        f = None
    basis = compute_basis(list(dv[..., t_ind] for dv in dvs_all))
    proj_all = []
    for pop_p, fvs_p in pairs:
        pop_pt = pop_p[..., t_ind]
        proj = np.swapaxes(basis, -1, -2) @ pop_pt
        proj = np.mean(proj, axis=0).T
        proj = average_similar_stimuli(proj, fvs_p, radius=radius)
        proj_all.append(proj)
    p = skd.PCA(3)
    p.fit_transform(np.concatenate(proj_all, axis=0))
    for i, proj in enumerate(proj_all):
        pts = p.transform(proj)
        ax.scatter(
            *pts.T,
            c=pairs[i][1][:, fv_ind],
            cmap=color_maps[i],
            s=ms,
            vmin=vmin,
            vmax=vmax,
        )
    ax.set_aspect("equal")
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    ax.set_zlabel("dim 3")
    return f, ax


@gpl.ax_adder()
def feature_projection_scatter_color(
    ref,
    new,
    ref_ind=-1,
    new_ind=0,
    t_ind=0,
    dv_ind=0,
    choice_ind=0,
    corr_ind=1,
    inc_frac=2,
    pt_size=2,
    outline_ind=0,
    ax=None,
):
    choice_dim = ref["dvs"][ref_ind][..., dv_ind, t_ind]
    pop_i = new["pops"][new_ind][..., t_ind]

    proj = choice_dim @ pop_i
    info = new["trial_info"][new_ind].to_numpy()
    outline = info[:, outline_ind]
    feats = new["feats"][new_ind]
    bound = np.max(np.abs(proj))
    ax.scatter(
        *feats[:, 1:].T,
        c=outline,
        cmap="bwr",
        s=pt_size * inc_frac,
    )
    ax.scatter(
        *feats[:, 1:].T,
        c=np.mean(proj, axis=0),
        cmap="bwr",
        vmin=-bound,
        vmax=bound,
        s=pt_size,
    )
    ax.set_aspect("equal")
    gpl.clean_plot(ax, 1)
    gpl.clean_plot_bottom(ax)


@gpl.ax_adder()
def feature_projection_hist(
    ref,
    new,
    ref_ind=-1,
    new_ind=0,
    t_ind=0,
    dv_ind=0,
    choice_ind=0,
    corr_ind=1,
    inc_frac=2,
    pt_size=2,
    outline_ind=0,
    ax=None,
    n_trls=5,
):
    choice_dim = ref["dvs"][ref_ind][..., dv_ind, t_ind]
    pop_i = new["pops"][new_ind][..., t_ind]

    proj = choice_dim @ pop_i
    info = new["trial_info"][new_ind].to_numpy()
    outline = info[:, outline_ind]

    ax.hist(np.mean(proj, axis=0)[outline == 1][:n_trls], density=True)
    ax.hist(np.mean(proj, axis=0)[outline == 2][:n_trls], density=True)


@gpl.ax_adder()
def feature_projection_tc(
    ref,
    new,
    ref_ind=-1,
    new_ind=0,
    t_ind=0,
    dv_ind=0,
    choice_ind=0,
    cat_ind=1,
    inc_frac=2,
    pt_size=2,
    ax=None,
    n_trls=5,
    trl_win=100,
):
    ests = ref["estimators"][ref_ind][..., t_ind]
    pop_i = new["pops"][new_ind][..., t_ind]

    preds = np.stack(list(est.predict(pop_i.T) for est in ests), axis=0)
    info = new["trial_info"][new_ind].to_numpy()
    choice = info[:, choice_ind] > 1.5
    pred = preds[..., dv_ind] == choice[None]
    corr_cat = info[:, cat_ind] > 1.5
    corr_choice = np.expand_dims(choice == corr_cat, 0)
    pred_corr = preds[..., dv_ind] == corr_cat[None]

    filt = np.ones((1, trl_win))
    pred_filt = ssig.convolve(pred, filt / np.sum(filt), mode="valid")

    corr_filt = ssig.convolve(corr_choice, filt / np.sum(filt), mode="valid")

    pcorr_filt = ssig.convolve(pred_corr, filt / np.sum(filt), mode="valid")

    trl = np.arange(pred_filt.shape[1])
    gpl.plot_trace_werr(trl, corr_filt, ax=ax, label="correct choice")
    gpl.plot_trace_werr(trl, pcorr_filt, ax=ax, label="correct prediction from ref")
    gpl.plot_trace_werr(trl, pred_filt, ax=ax, confstd=True, label="choice follows ref")

    gpl.add_hlines(0.5, ax)


@gpl.ax_adder()
def choice_projections(
    dv_reference,
    target_activity,
    choice,
    t_ind=0,
    use_dim=0,
    ax=None,
    plot_points=True,
    n_bins=6,
    rescale_choice=True,
    **kwargs,
):
    if rescale_choice:
        choice = 2 * (choice - 1.5)
    dv_use = dv_reference[..., use_dim, t_ind]
    act = target_activity[..., t_ind]
    proj = dv_use @ act
    m = sklm.LogisticRegression()
    m.fit(np.expand_dims(np.mean(proj, axis=0), 1), choice)
    lr = m.coef_
    if plot_points:
        ax.plot(np.mean(proj, axis=0), choice, "o", **kwargs)
    gpl.plot_scatter_average(
        np.mean(proj, axis=0), choice, ax=ax, n_bins=n_bins, **kwargs
    )
    ax.set_xlabel("projection onto decision axis")
    ax.set_ylabel("average choice")
    return lr


@gpl.ax_adder()
def plot_choice_corr(
    dvs,
    ref_dv=None,
    session_ts=None,
    choice_ind=0,
    opt_ind=1,
    ref_ind=0,
    t_ind=0,
    ax=None,
    **kwargs,
):
    corrs = np.zeros((dvs[0].shape[0], len(dvs)))
    refs = np.zeros_like(corrs)
    refs_opt = np.zeros_like(corrs)

    if session_ts is None:
        session_ts = np.arange(len(dvs))
    for i, dv in enumerate(dvs):
        dv_t = dv[..., t_ind]
        cv = u.make_unit_vector(dv_t[..., choice_ind])
        ov = u.make_unit_vector(dv_t[..., opt_ind])
        corrs[:, i] = np.sum(cv * ov, axis=1)
        if ref_dv is not None:
            rd = u.make_unit_vector(ref_dv[..., ref_ind, t_ind])
            refs[:, i] = np.sum(cv * rd, axis=1)
            refs_opt[:, i] = np.sum(ov * rd, axis=1)
    gpl.plot_trace_werr(session_ts, corrs, ax=ax, confstd=True, **kwargs)
    if ref_dv is not None:
        _ = kwargs.pop("label", None)
        gpl.plot_trace_werr(
            session_ts, refs, ax=ax, confstd=True, label="ref", **kwargs
        )
        gpl.plot_trace_werr(
            session_ts, refs_opt, ax=ax, confstd=True, label="ref opt", **kwargs
        )
    gpl.add_hlines(0, ax)


@gpl.ax_adder()
def plot_dv_corr(
    dvs,
    use_ind=0,
    t_ind=0,
    session_ts=None,
    ax=None,
    color_bounds=(0.2, 0.99),
    cmap="Blues",
):
    if session_ts is None:
        session_ts = np.arange(len(dvs))
    corrs = np.zeros((len(dvs), len(dvs), dvs[0].shape[0]))
    for i, dv_i in enumerate(dvs):
        for j, dv_j in enumerate(dvs):
            if i == j:
                corrs[i, j] = np.nan
            else:
                x_i = u.make_unit_vector(dv_i[..., use_ind, t_ind])
                x_j = u.make_unit_vector(dv_j[..., use_ind, t_ind])
                corrs[i, j] = np.sum(x_i * x_j, axis=1)
    colors = plt.get_cmap(cmap)(np.linspace(*color_bounds, len(corrs)))
    for i, corrs_i in enumerate(corrs):
        gpl.plot_trace_werr(session_ts, corrs_i.T, ax=ax, confstd=True, color=colors[i])
    gpl.add_hlines(0, ax)


@gpl.axs_adder(1, 3)
def choice_projections_all(
    dv_reference,
    activity,
    choices,
    axs=None,
    cmap="Blues",
    cmap_corr="Reds",
    color_bounds=(0.2, 0.9),
    bhv_ind=0,
    corr_ind=1,
    plot_points=False,
    **kwargs,
):
    cm = plt.get_cmap(cmap)
    col_num = np.linspace(*color_bounds, len(activity))
    colors = cm(col_num)
    cm_corr = plt.get_cmap(cmap_corr)
    colors_corr = cm_corr(col_num)
    lrs = []
    lr_corrs = []
    for i, act_i in enumerate(activity):
        lr, _ = choice_projections(
            dv_reference,
            act_i,
            choices[i].to_numpy()[:, bhv_ind],
            color=colors[i],
            ax=axs[0],
            plot_points=plot_points,
            fill=False,
            zorder=-i,
        )
        lrs.append(lr)

        lr_corr, _ = choice_projections(
            dv_reference,
            act_i,
            choices[i].to_numpy()[:, corr_ind],
            color=colors_corr[i],
            ax=axs[1],
            plot_points=plot_points,
            fill=False,
            zorder=-i,
        )
        lr_corrs.append(np.squeeze(lr_corr))

    axs[0].set_title("choice")
    axs[1].set_title("correct")
    lrs = np.squeeze(np.concatenate(lrs))
    xs = np.arange(len(activity))
    gpl.plot_colored_line(xs, lrs, cmap=cmap, ax=axs[2])
    gpl.plot_colored_line(xs, lr_corrs, cmap=cmap_corr, ax=axs[2])
    gpl.clean_plot(axs[2], 0)
    gpl.add_hlines(0, axs[2])
    axs[2].set_xlabel("session number")
    axs[2].set_ylabel("correlation")


@gpl.ax_adder()
def plot_shape_transition_performance(
    data_dict,
    ax=None,
    corr_field="correct",
    filt_wid=10,
    filt_step=10,
    plot_len=100,
    gap=20,
    cmaps=None,
    cbound_begin=(0.3, 0.55),
    cbound_end=(0.65, 0.99),
):
    if cmaps is None:
        cmaps = (None,) * len(data_dict)
    base = 0
    for i, (shape, d_use) in enumerate(data_dict.items()):
        corr_traj = np.concatenate(d_use[corr_field], axis=0)
        filt = np.ones(filt_wid) / filt_wid
        corr_filt = np.convolve(corr_traj, filt, mode="valid")
        xs_begin = np.arange(plot_len) + base
        xs_end = np.arange(plot_len) + plot_len + base + gap
        base = base + 2 * plot_len + base + gap
        gpl.plot_colored_line(
            xs_begin,
            corr_filt[:plot_len],
            cmap=cmaps[i],
            ax=ax,
            color_bounds=cbound_begin,
        )
        gpl.plot_colored_line(
            xs_end, corr_filt[-plot_len:], cmap=cmaps[i], ax=ax, color_bounds=cbound_end
        )
    gpl.add_hlines(0.5, ax)


@gpl.ax_adder()
def plot_cross_session_performance(
    data_dict,
    cmaps=None,
    corr_field="correct",
    filt_wid=200,
    ax=None,
):
    base = 0
    if cmaps is None:
        cmaps = (None,) * len(data_dict)
    for i, (shape, d_use) in enumerate(data_dict.items()):
        trls = np.concatenate(d_use[corr_field], axis=0)
        filt = np.ones(filt_wid) / filt_wid
        smooth_corr = np.convolve(trls, filt, mode="valid")
        trl_inds = np.arange(len(smooth_corr)) + base
        gpl.plot_colored_line(trl_inds, smooth_corr, cmap=cmaps[i], ax=ax)
        base = base + len(trl_inds)
    gpl.add_hlines(0.5, ax)
    ax.set_xlabel("trial number")
    ax.set_ylabel("fraction correct")
    gpl.clean_plot(ax, 0)
    return ax


def project_features(
    dvs,
    test_inds,
    orig_pair,
    *pairs,
    axs=None,
    fwid=3,
    cmap="magma",
    t_ind=0,
    ms=1,
    fv_ind=0,
    cmap_proj="magma",
    plot_original=True,
):
    pop, fvs = orig_pair
    if axs is None:
        f, axs = plt.subplots(
            len(pairs),
            2,
            figsize=(2 * fwid, len(pairs) * fwid),
            sharex="col",
            sharey="col",
            squeeze=False,
        )
    qs = []
    for i, dv_i in enumerate(dvs):
        dv_i = dv_i[..., t_ind]
        q = compute_basis((dv_i,))
        qs.append(q)
        b1, b2 = dv_i.T @ q
        ax_fv, ax_proj = axs[0]
        ax_proj.plot([-b1[0], b1[0]], [-b1[1], b1[1]])
        ax_proj.plot([-b2[0], b2[0]], [-b2[1], b2[1]])
        if plot_original:
            inds = test_inds[i, t_ind]
            pop_t = pop[..., t_ind]
            proj = pop_t.T[inds] @ q
            ax_proj.scatter(*proj.T, c=fvs[inds, fv_ind], cmap=cmap, s=ms)
            ax_fv.scatter(*fvs[inds].T, c=fvs[inds, fv_ind], cmap=cmap, s=ms)

    for i, (pop_p, fvs_p) in enumerate(pairs):
        ax_fv, ax_proj = axs[i]
        pop_pt = pop_p[..., t_ind]
        proj_ps = []
        for q in qs:
            proj_p = pop_pt.T @ q
            proj_ps.append(proj_p)
        proj_p = np.mean(proj_ps, axis=0)
        ax_proj.scatter(*proj_p.T, c=fvs_p[:, fv_ind], cmap=cmap_proj, s=ms)
        ax_fv.scatter(*fvs_p.T, c=fvs_p[:, fv_ind], cmap=cmap_proj, s=ms)

    ax_proj.set_aspect("equal")
    ax_fv.set_aspect("equal")


def plot_performance(
    data,
    ax=None,
    cho_field="chosen_cat",
    targ_field="stim_sample_MAIN",
    trl_field="trial",
    day_field="day",
    color_gap=0.2,
    cm="Blues",
    **kwargs,
):
    if ax is None:
        f, ax = plt.subplots(1, 1)

    corr_groups = data[cho_field] == data[targ_field]
    trl_groups = data[trl_field]
    day_num = data[day_field].to_numpy(int)
    inds = np.argsort(day_num)
    cm = plt.get_cmap(cm)
    cols = cm(np.linspace(color_gap, 1 - color_gap, len(trl_groups)))
    for i, cg in enumerate(corr_groups):
        trls = trl_groups[i].to_numpy(float)
        trls[trls > 5000] = np.nan
        col_i = cols[inds[i]]

        gpl.plot_scatter_average(trls, cg.to_numpy(float), **kwargs, ax=ax, color=col_i)
    return ax


@gpl.ax_adder
def plot_cross_shape_generalization(
    shape,
    pre_session,
    post_session,
    ax=None,
    fwid=1,
    shape_color=None,
    session_color=None,
    gen_tick=0.2,
    t_ind=0,
    key_pairs=(("dec", "gen"), ("dec_flip", "gen_flip")),
    markers=("o", "s"),
):
    out_list = (pre_session, shape, post_session)
    colors = (session_color, shape_color, session_color)
    for i, ol in enumerate(out_list):
        xs = [i - gen_tick / 2, i + gen_tick / 2]
        for j, (k1, k2) in enumerate(key_pairs):
            pts_k1 = ol[k1][..., t_ind]
            pts_k2 = ol[k2][..., t_ind]
            pts = np.stack((pts_k1, pts_k2), axis=-1)
            for pt in pts:
                gpl.plot_trace_werr(
                    xs,
                    pt,
                    ax=ax,
                    points=True,
                    confstd=True,
                    color=colors[i],
                    fill=False,
                    marker=markers[j],
                )
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["pre", "shape", "post"])
    gpl.add_hlines(0.5, ax)


def plot_eg_neurons(pop, xs, axs=None, fwid=1):
    if axs is None:
        n_plot = int(np.ceil(np.sqrt(pop.shape[1])))
        f, axs = plt.subplots(n_plot, n_plot, figsize=(fwid * n_plot, fwid * n_plot))
        axs = axs.flatten()
    for i in range(pop.shape[1]):
        gpl.plot_trace_werr(xs, pop[:, i], ax=axs[i])
    return axs


def plot_decoding_tc(dec, xs, gen=None, axs=None, ax=None, fwid=3, time_key="stim on"):
    n_plots = len(dec)
    if axs is None and ax is None:
        f, axs = plt.subplots(n_plots, 1, figsize=(fwid, n_plots * fwid))
    if axs is None and ax is not None:
        axs = np.zeros((n_plots), dtype=object)
        axs[:] = ax

    for i, (k, dec_i) in enumerate(dec.items()):
        ls = gpl.plot_trace_werr(
            xs,
            dec_i,
            ax=axs[i],
            confstd=True,
        )
        if gen is not None:
            col = ls[0].get_color()
            gpl.plot_trace_werr(
                xs,
                gen[i],
                color=col,
                ax=axs[i],
                ls="dashed",
                confstd=True,
            )
        axs[i].set_ylabel("decoding performance")
        gpl.add_hlines(0.5, axs[i])
    axs[i].set_xlabel("time relative to {}".format(time_key))
    return axs


def plot_decoding_hist(
    dec_dict,
    xs,
    t_ind=200,
    ax=None,
    time_key="stim on",
    only_region=None,
    color=None,
    cmap=None,
    col_cut=0.3,
    x_range=None,
    **kwargs,
):
    if ax is None:
        f, ax = plt.subplots(1, 1)
    out_sd = _sort_decs(dec_dict, xs, t_ind, only_region=only_region)
    days, shapes = out_sd[:2]

    preds, targs = out_sd[-2:]
    if cmap is not None:
        cm = plt.get_cmap(cmap)
        colors = cm(np.linspace(col_cut, 1 - col_cut, len(days)))
    else:
        colors = (color,) * len(days)
    x_min, x_max = np.inf, -np.inf
    for i, pred in enumerate(preds):
        xs = np.concatenate(list(p_j.flatten() for p_j in pred))
        if len(xs) > 0:
            ax.hist(xs, color=colors[i], density=True, histtype="step", **kwargs)
            x_min = np.min([np.min(xs), x_min])
            x_max = np.max([np.max(xs), x_max])
    if x_range is None:
        ax.set_xlim([x_min, x_max])
    else:
        ax.set_xlim(x_range)
    return ax


def plot_decoding_scatter(
    dec_dict,
    xs,
    t_ind=200,
    ax=None,
    time_key="stim on",
    only_region=None,
    color=None,
    cmap=None,
    col_cut=0.3,
    ms=1,
    y_range=None,
    x_range=None,
    use_targ_range=True,
    **kwargs,
):
    if ax is None:
        f, ax = plt.subplots(1, 1)
    out_sd = _sort_decs(dec_dict, xs, t_ind, only_region=only_region)
    days, shapes = out_sd[:2]

    preds, targs = out_sd[-2:]
    if cmap is not None:
        cm = plt.get_cmap(cmap)
        colors = cm(np.linspace(col_cut, 1 - col_cut, len(days)))
    else:
        colors = (color,) * len(days)
    y_min, y_max = np.inf, -np.inf
    x_min, x_max = np.inf, -np.inf
    for i, pred in enumerate(preds):
        targ = targs[i]
        xs = np.concatenate(list(p_j.flatten() for p_j in pred))
        ys = np.concatenate(list(t_j.flatten() for t_j in targ))
        if len(xs) > 0:
            ax.plot(xs, ys, "o", ms=ms, color=colors[i], rasterized=True)
            y_min = np.min([np.min(ys), y_min])
            y_max = np.max([np.max(ys), y_max])
            x_min = np.min([np.min(xs), x_min])
            x_max = np.max([np.max(xs), x_max])
    if use_targ_range:
        x_min, x_max = y_min, y_max
    if y_range is None:
        ax.set_ylim([y_min, y_max])
    else:
        ax.set_ylim(y_range)
    if x_range is None:
        ax.set_xlim([x_min, x_max])
    else:
        ax.set_xlim(x_range)
    ax.plot([x_min, x_max], [y_min, y_max], color=(0.8, 0.8, 0.8))
    return ax


def _sort_decs(
    dec_dict,
    xs,
    t_ind=200,
    only_region=None,
):
    if t_ind is not None:
        x_ind = np.argmin((t_ind - xs) ** 2)
    days = []
    shapes = []
    for i, ((day, shape, region), v) in enumerate(dec_dict.items()):
        dec = v[0]
        if i == 0:
            store = tuple([] for i in v)
        comp = not np.any(np.isnan(dec))
        if comp and (only_region is None or only_region == region):
            days.append(day)
            shapes.append(shape)
            if t_ind is not None:
                v = list(v_j[..., x_ind] for v_j in v)
            list(s_j.append(v[j]) for j, s_j in enumerate(store))
    outs = []
    for s_j in store:
        try:
            s_j_comb = np.stack(s_j, axis=0)
        except ValueError:
            s_j_comb = np.zeros(len(s_j), dtype=object)
            for k, s_jk in enumerate(s_j):
                s_j_comb[k] = s_j[k]
        outs.append(s_j_comb)
    outs = tuple(outs)
    if len(days) > 0:
        days = np.stack(days, axis=0)
        shapes = np.stack(shapes, axis=0)
        sort_inds = np.argsort(days)
        days = days[sort_inds]
        shapes = shapes[sort_inds]
        outs = tuple(o[sort_inds] for o in outs)
    return (
        days,
        shapes,
    ) + outs


def plot_latent_space(lv1, lv2, imgs, axs=None, fwid=2):
    lv1_u = np.unique(lv1)
    lv2_u = np.unique(lv2)
    if axs is None:
        f, axs = plt.subplots(
            len(lv1_u), len(lv2_u), figsize=(fwid * len(lv2_u), fwid * len(lv1_u))
        )
    img_dict = {(lv1_i, lv2[i]): imgs[i] for i, lv1_i in enumerate(lv1)}
    for i, j in it.product(range(len(lv1_u)), range(len(lv2_u))):
        img_ij = img_dict.get((lv1_u[i], lv2_u[j]))
        if img_ij is not None:
            axs[i, j].imshow(img_ij)
            axs[i, j].set_aspect("equal")
        gpl.clean_plot(axs[i, j], 1)
        gpl.clean_plot_bottom(axs[i, j])
    return axs


def plot_decoding_dict_tc(
    dec_dict,
    xs,
    ax=None,
    fwid=3,
    time_key="stim on",
    only_region=None,
    plot_gen=False,
    color=None,
    cmap=None,
    col_cut=0.3,
    label="",
    **kwargs,
):
    if ax is None:
        f, ax = plt.subplots(1, 1)
    out = _sort_decs(dec_dict, xs, None, only_region=only_region)
    days, shapes, decs = out[:3]
    gens = out[-1]
    if cmap is not None:
        cm = plt.get_cmap(cmap)
        colors = cm(np.linspace(col_cut, 1 - col_cut, len(days)))
    else:
        colors = (color,) * len(days)

    if plot_gen:
        decs = gens

    labels = ("",) * (len(decs) - 1) + (label,)
    for i, dec_i in enumerate(decs):
        gpl.plot_trace_werr(
            xs,
            np.mean(dec_i, axis=0),
            ax=ax,
            lw=0.8,
            color=colors[i],
            label=labels[i],
            **kwargs,
        )

    ax.set_xlabel("time from {}".format(time_key))


def plot_aggregate_session_dec(
    dec_dict,
    xs,
    t_ind=200,
    pt=0,
    ax=None,
    plot_gen=False,
    only_region=None,
    **kwargs,
):
    if ax is None:
        f, ax = plt.subplots(1, 1)

    days, shapes, decs, gens = _sort_decs(dec_dict, xs, t_ind, only_region=only_region)
    if plot_gen:
        decs = gens
    decs = np.mean(decs, axis=1)
    gpl.violinplot([decs], [pt], ax=ax, **kwargs)


def plot_cross_session_dec(
    dec_dict,
    xs,
    t_ind=200,
    ax=None,
    plot_gen=False,
    only_region=None,
    **kwargs,
):
    if ax is None:
        f, ax = plt.subplots(1, 1)
    days, shapes, decs, gens = _sort_decs(dec_dict, xs, t_ind, only_region=only_region)
    if plot_gen:
        decs = gens

    gpl.plot_trace_werr(days, decs.T, ax=ax, conf95=True, **kwargs)
    return ax


def plot_session_rfs(feats, pop, min_trls=5, axs=None, fwid=1, cmap="Blues"):
    if axs is None:
        n_plots = int(np.ceil(np.sqrt(pop.shape[1])))
        f, axs = plt.subplots(
            n_plots, n_plots, figsize=(fwid * n_plots, fwid * n_plots)
        )
        axs = axs.flatten()
    ns, (e1, e2) = np.histogramdd(feats)
    xs = e1[:-1] + np.diff(e1)[0] / 2
    ys = e2[:-1] + np.diff(e2)[0] / 2
    for i in range(pop.shape[1]):
        spks, (e1, e2) = np.histogramdd(feats, weights=pop[:, i])
        img = spks / ns
        img[ns < min_trls] = np.nan
        gpl.pcolormesh(xs, ys, img, cmap="Blues", ax=axs[i])
        gpl.clean_plot(axs[i], 1)
        gpl.clean_plot_bottom(axs[i])
    return axs


@gpl.ax_adder()
def plot_decoder_autocorrelation_map(
    shapes,
    dates,
    xs,
    gen_arr,
    ax=None,
    use_cmaps=("Blues", "Reds", "Greens", "Purples", "Oranges"),
    heat_cmap="Greys",
    t_targ=250,
    chance=0.5,
    normalize=False,
    c_use=0.8,
    fwid=3,
    y_label="",
):
    shapes = np.array(shapes)
    t_ind = np.argmin(np.abs(xs - t_targ))
    dates = slaux.parse_dates(dates)
    _, inds = np.unique(shapes, return_index=True)
    u_shapes = shapes[np.sort(inds)]

    shapes_nice = list(s.strip("None").split(".") for s in u_shapes)

    set_colors = {}
    cmaps = []
    for i, sn in enumerate(shapes_nice):
        s = sn[0]
        cmap = set_colors.get(s, use_cmaps[len(set_colors) % len(use_cmaps)])
        set_colors[s] = cmap
        cmaps.append(plt.get_cmap(cmap))

    colors = list(cm(c_use) for cm in cmaps)
    plot_arr = np.zeros(gen_arr.shape)
    for i, j in u.make_array_ind_iterator(gen_arr.shape):
        y = gen_arr[i, j]
        if normalize:
            y = y / gen_arr[i, i]
        y_mu = np.mean(y[:, t_ind])
        plot_arr[i, j] = y_mu
    days = (dates - min(dates)).days
    u_ticks = np.arange(len(dates))
    gpl.pcolormesh(u_ticks, u_ticks, plot_arr, cmap=heat_cmap, ax=ax)
    ax.set_xticks(u_ticks[::10])
    ax.set_xticklabels(days[::10])
    ax.set_yticks(u_ticks[::10])
    ax.set_yticklabels(days[::10])
    for i, us in enumerate(u_shapes):
        match_inds = np.where(us == shapes)[0]
        start, end = match_inds[0], match_inds[-1]
        ax.hlines(-1, u_ticks[start], u_ticks[end], color=colors[i])
        ax.vlines(u_ticks[-1] + 1, u_ticks[start], u_ticks[end], color=colors[i])
    gpl.clean_plot(ax, 0)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.invert_yaxis()
    ax.set_xlabel("trained day")
    ax.set_ylabel("tested day")
    ax.set_aspect("equal")


def plot_decoder_autocorrelation_full(
    shapes,
    dates,
    xs,
    gen_arr,
    axs=None,
    use_cmaps=("Blues", "Reds", "Greens", "Purples", "Oranges"),
    t_targ=250,
    chance=0.5,
    normalize=False,
    c_range=(0.3, 0.9),
    fwid=3,
    y_label="generalization performance",
):
    shapes = np.array(shapes)
    t_ind = np.argmin(np.abs(xs - t_targ))
    dates = slaux.parse_dates(dates)
    _, inds = np.unique(shapes, return_index=True)
    u_shapes = shapes[np.sort(inds)]
    shapes_nice = list(s.strip("None").split(".") for s in u_shapes)

    set_colors = {}
    cmaps = []
    for i, sn in enumerate(shapes_nice):
        s = sn[0]
        cmap = set_colors.get(s, use_cmaps[len(set_colors) % len(use_cmaps)])
        set_colors[s] = cmap
        cmaps.append(plt.get_cmap(cmap))
    if axs is None:
        n_sh = len(u_shapes)
        f, axs = plt.subplots(
            n_sh, n_sh, figsize=(n_sh * fwid, n_sh * fwid), sharey=True
        )
    else:
        f = None
        n_sh = axs.shape[0]
    for i, j in it.product(range(n_sh), repeat=2):
        sh_i = u_shapes[i]
        sh_j = u_shapes[j]

        if i == 0:
            axs[i, j].set_title(shapes_nice[j][0])
        row_mask = shapes == sh_i
        col_mask = shapes == sh_j
        sub_arr = gen_arr[row_mask][:, col_mask]
        sub_arr_ii = gen_arr[row_mask][:, row_mask]
        dates_i = dates[row_mask]
        dates_j = dates[col_mask]
        min_date = min(dates_j)
        num_days = (max(dates_j) - min_date).days + 1
        for ai, aj in u.make_array_ind_iterator(sub_arr.shape):
            x = (dates_j[aj] - dates_i[ai]).days
            date_prog = (dates_j[aj] - min_date).days / num_days

            date_prog_norm = date_prog * (c_range[1] - c_range[0]) + c_range[0]
            color = cmaps[j](date_prog_norm)
            y = sub_arr[ai, aj]
            if normalize:
                y = y / sub_arr_ii[ai, ai]
            gpl.plot_trace_werr(
                [x],
                np.expand_dims(y[:, t_ind], 1),
                ax=axs[i, j],
                fill=False,
                color=color,
                confstd=True,
                points=True,
            )
        if chance is not None:
            gpl.add_hlines(chance, axs[i, j])
        axs[i, j].set_xlabel("day difference")
        if j == 0:
            axs[i, j].set_ylabel(y_label)
        gpl.clean_plot(axs[i, j], j)
    return f, axs


@gpl.ax_adder()
def plot_decoder_autocorrelation(
    shapes,
    dates,
    xs,
    gen_arr,
    ax=None,
    cmap="coolwarm",
    same_cm=0.99,
    first_cm=0,
    second_cm=0.4,
    t_targ=250,
    chance=0.5,
    normalize=False,
):
    cm = plt.get_cmap(cmap)
    t_ind = np.argmin(np.abs(xs - t_targ))
    dates = slaux.parse_dates(dates)
    for i, j in u.make_array_ind_iterator(gen_arr.shape):
        x = (dates[j] - dates[i]).days
        y = gen_arr[i, j]
        if normalize:
            y = y / gen_arr[i, i]
        s_i = shapes[i].strip("None").split(".")
        s_j = shapes[j].strip("None").split(".")

        if s_i[0] == s_j[0]:
            color = cm(same_cm)
        elif len(s_i) == 1 and len(s_j) == 1:
            color = cm(first_cm)
        else:
            color = cm(second_cm)
        gpl.plot_trace_werr(
            [x],
            np.expand_dims(y[:, t_ind], 1),
            ax=ax,
            fill=False,
            color=color,
            confstd=True,
            points=True,
        )
    gpl.add_hlines(chance, ax)
    ax.set_xlabel("day difference")
    ax.set_ylabel("generalization performance")


def plot_nl_decision_boundary(m, var_range=(-1, 1), ax=None, n_pts=100, cmap="bwr"):
    pts = np.linspace(*var_range, n_pts)
    xv, yv = np.meshgrid(pts, pts)
    xv_flat = np.reshape(xv, (-1, 1))
    yv_flat = np.reshape(yv, (-1, 1))
    preds = m.decision_function(np.concatenate((xv_flat, yv_flat), axis=1))
    dec_map = np.reshape(preds, xv.shape)

    cm = plt.get_cmap(cmap)
    gpl.pcolormesh(pts, pts, dec_map, ax=ax, cmap=cm)


def plot_sampled_stimuli(
    data,
    ind=0,
    ax=None,
    cat_bound_field="cat_def_MAIN",
    stim_cat_field="stim_sample_MAIN",
    stim_feat_field="stim_feature_MAIN",
    color1="r",
    color2="b",
    ms=1,
    plot_effective_boundary=True,
):
    if ax is None:
        f, ax = plt.subplots(1, 1)

    cats = data[stim_cat_field][ind].to_numpy()
    feats = np.stack(data[stim_feat_field][ind].to_numpy(), axis=0) / 1000

    ax.plot(feats[cats == 1][:, 0], feats[cats == 1][:, 1], "o", color=color1, ms=ms)
    ax.plot(feats[cats == 2][:, 0], feats[cats == 2][:, 1], "o", color=color2, ms=ms)

    cat_boundary_angles = np.unique(data[cat_bound_field][ind].to_numpy())
    for cba in cat_boundary_angles:
        ax.plot(
            np.cos(np.radians(cba)) * np.array([-1, +1]),
            np.sin(np.radians(cba)) * np.array([-1, +1]),
            "k",
        )
    if plot_effective_boundary:
        eb = sla.estimate_decision_boundary(data, ind=ind)
        ax.plot(
            eb["boundary"][0][0] * np.array([-1, +1]),
            -eb["boundary"][0][1] * np.array([-1, +1]),
            "k",
            linestyle="dashed",
        )

    gpl.clean_plot(ax, 0)
    gpl.make_xaxis_scale_bar(ax, magnitude=0.5, label="width", text_buff=0.28)
    gpl.make_yaxis_scale_bar(ax, magnitude=0.5, label="height", text_buff=0.45)
