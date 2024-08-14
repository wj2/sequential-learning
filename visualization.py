import matplotlib.pyplot as plt
import numpy as np
import itertools as it

import general.plotting as gpl
import general.utility as u
import sequential_learning.analysis as sla
import sequential_learning.auxiliary as slaux


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
