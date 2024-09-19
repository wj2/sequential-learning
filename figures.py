import os
import numpy as np
import matplotlib.pyplot as plt

import general.data_io as gio
import general.paper_utilities as pu
import general.utility as u
import general.plotting as gpl
import sequential_learning.auxiliary as slaux
import sequential_learning.analysis as sla
import sequential_learning.visualization as slv

config_path = "sequential_learning/figures.conf"
colors = (
    np.array(
        [
            (127, 205, 187),
            (65, 182, 196),
            (29, 145, 192),
            (34, 94, 168),
            (37, 52, 148),
            (8, 29, 88),
        ]
    )
    / 256
)


class SequenceLearningFigure(pu.Figure):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, find_panel_keys=False, **kwargs)

    def only_load_shape_data(self, shape="A6", max_files=np.inf, validate=True):
        data = gio.Dataset.from_readfunc(
            slaux.load_kiani_data_folder,
            os.path.join(self.params.get("data_folder"), shape),
            max_files=max_files,
            sort_by="day",
        )
        if validate:
            data = slaux.filter_valid(data)
        return data

    def load_shape_data(self, *args, **kwargs):
        if self.data.get("exper_data") is None:
            data = self.only_load_shape_data(*args, **kwargs)
            self.data["exper_data"] = data
        return self.data["exper_data"]

    def load_shape_groups(self):
        if self.data.get("exper_data") is None:
            data_dict = slaux.load_shape_list(self.shape_sequence)
            self.data["exper_data"] = data_dict
        return self.data["exper_data"]

    @property
    def color_dict(self):
        regions = self.params.getlist("use_regions")
        color_dict = {r: self.params.getcolor("{}_color".format(r)) for r in regions}
        return color_dict

    @property
    def resp_colors(self):
        r1 = self.params.getcolor("r1_color")
        r2 = self.params.getcolor("r2_color")
        return r1, r2

    @property
    def cm_dict(self):
        regions = self.params.getlist("use_regions")
        color_dict = {r: self.params.get("{}_cm".format(r)) for r in regions}
        return color_dict

    def _generic_decoding(
        self,
        key,
        func_and_name,
        plot_gen=None,
        chance=0.5,
        recompute=False,
        inset=False,
        scatter_bound=(-0.75, 0.75),
        strict_prototype=False,
    ):
        if plot_gen is None:
            plot_gen = {}
        if inset:
            axs, axs_inset = self.gss[key]
        else:
            axs = self.gss[key]

        var_thr = self.params.getfloat("sample_var_ratio")
        regions = self.params.getlist("use_regions")
        winsize = self.params.getint("winsize")
        tbeg = self.params.getint("tbeg")
        tend = self.params.getint("tend")
        step = self.params.getint("winstep")
        uniform_resample = self.params.getboolean("uniform_resample")
        if self.data.get(key) is None or recompute:
            data = self.load_shape_data()
            if strict_prototype:
                m1, m2 = slaux.get_strict_prototype_masks(data)[0]
                data_session = data.mask(m1.rs_or(m2))
                uniform_resample = False
            else:
                var_ratio = sla.compute_var_ratio(data)
                data_session = data.session_mask(var_ratio > var_thr)
            outs = {}
            for r in regions:
                args = (data_session, winsize, tbeg, tend, step)
                kwargs_both = {
                    "regions": (r,),
                    "uniform_resample": uniform_resample,
                    "estimate_performance": True,
                }
                kwargs_stim = {}
                kwargs_sacc = {"time_zero_field": "fp_off"}
                for k, func in func_and_name.items():
                    out_stim = func(*args, **kwargs_stim, **kwargs_both)
                    out_sacc = func(*args, **kwargs_sacc, **kwargs_both)
                    store = outs.get(k, {})
                    store[r] = (out_stim, out_sacc)
                    outs[k] = store
            self.data[key] = outs

        outs_all = self.data[key]
        for i, r in enumerate(regions):
            for j, (k, out_ij) in enumerate(outs_all.items()):
                slv.plot_decoding_dict_tc(
                    *out_ij[r][0],
                    ax=axs[j, 0],
                    cmap=self.cm_dict[r],
                    plot_gen=plot_gen.get(k, False),
                )
                slv.plot_decoding_dict_tc(
                    *out_ij[r][1],
                    ax=axs[j, 1],
                    color=self.color_dict[r],
                    cmap=self.cm_dict[r],
                    time_key="fixation off",
                    plot_gen=plot_gen.get(k, False),
                    label=r,
                )
                if inset:
                    slv.plot_decoding_scatter(
                        *out_ij[r][0],
                        ax=axs_inset[j, 0],
                        cmap=self.cm_dict[r],
                        plot_gen=plot_gen.get(k, False),
                        x_range=scatter_bound,
                        y_range=scatter_bound,
                        ms=0.1,
                    )
                    slv.plot_decoding_scatter(
                        *out_ij[r][1],
                        ax=axs_inset[j, 1],
                        cmap=self.cm_dict[r],
                        plot_gen=plot_gen.get(k, False),
                        x_range=scatter_bound,
                        y_range=scatter_bound,
                        ms=0.1,
                    )
                axs[j, 0].set_ylabel(k)
                if j < len(outs_all) - 1:
                    axs[j, 0].set_xlabel("")
                    axs[j, 1].set_xlabel("")
                    gpl.clean_plot_bottom(axs[0, 0], keepticks=True)
                    gpl.clean_plot_bottom(axs[0, 1], keepticks=True)
                gpl.clean_plot(axs[j, 0], 0)
                gpl.clean_plot(axs[j, 1], 1)
                gpl.add_hlines(chance, axs[j, 0])
                gpl.add_hlines(chance, axs[j, 1])


class RelativeTransitionFigure(SequenceLearningFigure):
    def __init__(
        self,
        shapes=None,
        fig_key="relative_transition_figure",
        exper_data=None,
        region="IT",
        fig_folder="",
        uniform_resample=False,
        save_video=True,
        **kwargs,
    ):
        fsize = (14, 18)

        cf = u.ConfigParserColor()
        cf.read(config_path)
        params = cf[fig_key]
        self.fig_key = fig_key
        self.region = (region,)
        self.fig_folder = fig_folder
        self.uniform_resample = uniform_resample
        self.save_video = save_video
        if exper_data is not None:
            add_data = {"exper_data": exper_data}
            data = kwargs.get("data", {})
            data.update(add_data)
            kwargs["data"] = data
            self.shape_sequence = list(exper_data.keys())
        elif shapes is not None:
            self.shape_sequence = shapes
        else:
            raise IOError("either data or shape list must be provided")
        super().__init__(fsize, params, colors=colors, **kwargs)

    def make_gss(self):
        gss = {}

        task_grid = pu.make_mxn_gridspec(
            self.gs,
            1,
            2,
            0,
            20,
            0,
            60,
            5,
            5,
        )
        task_angs = pu.make_mxn_gridspec(
            self.gs,
            2,
            1,
            5,
            35,
            70,
            100,
            15,
            5,
        )
        task_axs = self.get_axs(task_grid, squeeze=True, sharex="all", sharey="all")
        angs_axs = self.get_axs(task_angs, squeeze=True, sharex="all")
        gss["panel_tasks"] = (task_axs, angs_axs[0])

        dec_ax = self.get_axs((self.gs[15:45, 0:60],), all_3d=True)[0, 0]
        gss["panel_dec"] = (dec_ax, angs_axs[1])

        learning_grid = pu.make_mxn_gridspec(self.gs, 2, 2, 45, 80, 0, 40, 4, 4)
        gss["panel_subspace"] = self.get_axs(
            learning_grid,
            sharey="all",
        )

        learning_grid = pu.make_mxn_gridspec(self.gs, 2, 3, 45, 80, 50, 100, 4, 4)
        gss["panel_learning"] = self.get_axs(learning_grid)

        gss["panel_bhv_learning"] = self.get_axs(
            (self.gs[85:100, 50:100],),
        )[0, 0]

        self.gss = gss

    def _analysis(
        self,
    ):
        data_dict = self.load_shape_groups()
        if self.data.get("main_analysis") is None:
            t_start = self.params.getfloat("t_start")
            t_end = self.params.getfloat("t_end")
            binsize = self.params.getfloat("binsize")
            binstep = self.params.getfloat("binstep")
            out = sla.joint_variable_shape_sequence(
                data_dict,
                regions=self.region,
                t_start=t_start,
                t_end=t_end,
                binsize=binsize,
                binstep=binstep,
                uniform_resample=self.uniform_resample,
                stim_field=["cat_proj", "anticat_proj"],
            )
            self.data["main_analysis"] = out
        return self.data["main_analysis"]

    def panel_tasks(self):
        key = "panel_tasks"
        t_axs, ang_ax = self.gss[key]

        sess_ind = -1
        data_dict = self.load_shape_groups()
        cmaps = self.params.getlist("colormaps")
        ang_vecs = []
        for i, (shape, d_use) in enumerate(data_dict.items()):
            ang = d_use["cat_def_MAIN"][sess_ind].iloc[0]

            f_both = np.stack(d_use["stim_feature_MAIN"][sess_ind], axis=0) / 1000
            cp_i = d_use["cat_proj"][sess_ind]
            t_axs[i].scatter(*f_both.T, c=cp_i, cmap=cmaps[i])
            ang_vecs.append(
                np.array((np.sin(np.radians(ang)), np.cos(np.radians(ang))))
            )
            t_axs[i].plot(
                [np.sin(np.radians(ang)), -np.sin(np.radians(ang))],
                [np.cos(np.radians(ang)), -np.cos(np.radians(ang))],
                color="k",
            )
            t_axs[i].plot(
                [np.cos(np.radians(ang)), 0],
                [-np.sin(np.radians(ang)), 0],
                color="r",
            )
            t_axs[i].set_aspect("equal")
            gpl.clean_plot(t_axs[i], 1)
            gpl.clean_plot_bottom(t_axs[i])
        gpl.add_vlines(ang_vecs[0] @ ang_vecs[1], ang_ax)

    def panel_dec(self):
        key = "panel_dec"
        vis_ax, ang_ax = self.gss[key]
        out = self._analysis()

        s1, s2 = self.shape_sequence
        s1_full_sampled = np.where(slaux.fully_sampled(out[s1]["feats"]))[0]
        s2_full_sampled = np.where(slaux.fully_sampled(out[s2]["feats"]))[0]

        color_maps = ("magma", "cividis")
        sess_ind_base_s1 = s1_full_sampled[-1]
        sess_ind_base_s2 = s2_full_sampled[0]
        sess_ind_s1 = (s1_full_sampled[-2],)
        sess_ind_s2 = (s2_full_sampled[1],)
        dv_s1 = out[s1]["dvs"][sess_ind_base_s1]
        dv_s2 = out[s2]["dvs"][sess_ind_base_s2]

        projs_s1 = list(
            (out[s1]["pops"][si], out[s1]["feats"][si]) for si in sess_ind_s1
        )
        projs_s2 = list(
            (out[s2]["pops"][si], out[s2]["feats"][si]) for si in sess_ind_s2
        )
        projs = projs_s1 + projs_s2

        _ = slv.project_features_common(
            (dv_s1, dv_s2),
            *projs,
            color_maps=color_maps,
            ax=vis_ax,
        )

        if self.save_video:
            f, ax = slv.project_features_common(
                (dv_s1, dv_s2),
                *projs,
                color_maps=color_maps,
            )
            fp = os.path.join(
                self.fig_folder, "vis_{}-{}.mp4".format(*self.shape_sequence),
            )
            gpl.rotate_3d_plot(f, ax, fp, fps=30)

        v_a2 = u.make_unit_vector(out[s1]["dvs"][-1][..., 0, 0])
        v_a2_i = u.make_unit_vector(out[s1]["dvs"][-2][..., 0, 0])
        v_a3 = u.make_unit_vector(out[s2]["dvs"][0][..., 0, 0])
        v_a3_i = u.make_unit_vector(out[s2]["dvs"][1][..., 0, 0])

        ang_ax.hist((v_a2 @ v_a3.T).flatten())
        ang_ax.hist((v_a2_i @ v_a2.T).flatten())
        ang_ax.hist((v_a3_i @ v_a3.T).flatten())

    def video_test(self):
        f, ax = plt.subplots(1, 1, subplot_kw={"projection": "3d"})
        fp = os.path.join(self.fig_folder, "vis_{}-{}.mp4".format(*self.shape_sequence))
        gpl.rotate_3d_plot(f, ax, fp, fps=30)

    def panel_subspace(self):
        key = "panel_subspace"
        axs = self.gss[key]
        out = self._analysis()
        s1, s2 = self.shape_sequence

        slv.plot_all_cross_projections(out[s1]["dvs"], out[s2]["dvs"], axs=axs)

    def panel_learning(self):
        key = "panel_learning"
        ax_cross, ax_within = self.gss[key]
        out = self._analysis()
        s1, s2 = self.shape_sequence

        dv_ref = out[s1]["dvs"][-1]
        pops = out[s2]["pops"]
        choices = out[s2]["trial_info"]
        slv.choice_projections_all(dv_ref, pops, choices, n_bins=6, axs=ax_cross)

        dv_ref = out[s2]["dvs"][-1]
        pops = out[s2]["pops"][:-1]
        choices = out[s2]["trial_info"][:-1]
        slv.choice_projections_all(dv_ref, pops, choices, n_bins=6, axs=ax_within)

    def panel_bhv_learning(self):
        key = "panel_bhv_learning"
        ax = self.gss[key]
        data_dict = self.load_shape_groups()
        slv.plot_cross_session_performance(data_dict, ax=ax, cmaps=("Blues", "Reds"))
        gpl.clean_plot(ax, 0)


class ShapeComparison(SequenceLearningFigure):
    def __init__(
        self,
        shape_sequence,
        fig_key="dec_sequence_figure",
        colors=colors,
        exper_data=None,
        fwid=3,
        **kwargs,
    ):
        fsize = (fwid * (len(shape_sequence) - 1), fwid)

        cf = u.ConfigParserColor()
        cf.read(config_path)
        params = cf[fig_key]
        self.fig_key = fig_key
        self.shape_sequence = shape_sequence
        if exper_data is not None:
            add_data = {"exper_data": exper_data}
            data = kwargs.get("data", {})
            data.update(add_data)
            kwargs["data"] = data
        super().__init__(fsize, params, colors=colors, **kwargs)

    def make_gss(self):
        gss = {}
        n_plots = len(self.shape_sequence) - 1
        dec_grid = pu.make_mxn_gridspec(self.gs, 1, n_plots, 0, 100, 0, 100, 2, 2)
        dec_ax = self.get_axs(
            dec_grid,
            squeeze=True,
            sharex="all",
            sharey="all",
        )
        gss["panel_dec"] = dec_ax

        self.gss = gss

    def panel_dec(self, recompute=True):
        key = "panel_dec"
        axs = self.gss[key]

        data_dict = self.load_shape_groups()
        session_color = self.params.getcolor("session_color")
        shape_color = self.params.getcolor("shape_color")

        if self.data.get(key) is None or recompute:
            gen_sequence = {}
            for i, s2 in enumerate(self.shape_sequence[1:]):
                s1 = self.shape_sequence[i]
                out = sla.compute_cross_shape_generalization(
                    data_dict[s1],
                    data_dict[s2],
                    500,
                    0,
                    500,
                    500,
                )
                gen_sequence[(s1, s2)] = out
            self.data[key] = gen_sequence
        gen_sequence = self.data[key]
        for i, ((s1, s2), gs_out) in enumerate(gen_sequence.items()):
            slv.plot_cross_shape_generalization(
                *gs_out, session_color=session_color, shape_color=shape_color, ax=axs[i]
            )
            axs[i].set_title("{} to {}".format(s1, s2))
            gpl.clean_plot(axs[i], i)


class BehaviorSequenceSummary(SequenceLearningFigure):
    def __init__(
        self,
        shape_sequence,
        fig_key="bhv_sequence_figure",
        colors=colors,
        exper_data=None,
        **kwargs,
    ):
        fsize = (16, 4)

        cf = u.ConfigParserColor()
        cf.read(config_path)
        params = cf[fig_key]
        self.fig_key = fig_key
        self.shape_sequence = shape_sequence
        if exper_data is not None:
            add_data = {"exper_data": exper_data}
            data = kwargs.get("data", {})
            data.update(add_data)
            kwargs["data"] = data
        super().__init__(fsize, params, colors=colors, **kwargs)

    def make_gss(self):
        n_shapes = len(self.shape_sequence)

        gss = {}
        learning_grid = pu.make_mxn_gridspec(
            self.gs,
            1,
            n_shapes,
            0,
            50,
            0,
            100,
            2,
            5,
        )
        learning_ax = self.get_axs(
            learning_grid,
            squeeze=True,
            sharey="all",
        )
        gss["panel_bhv"] = learning_ax.flatten()

        task_grid = pu.make_mxn_gridspec(
            self.gs,
            1,
            2 * n_shapes,
            55,
            100,
            0,
            100,
            2,
            2,
        )
        task_axs = self.get_axs(
            task_grid,
            squeeze=True,
            sharex="all",
            sharey="all",
        )
        gss["panel_tasks"] = np.reshape(task_axs, (-1, 2))

        self.gss = gss

    def panel_bhv(self):
        key = "panel_bhv"
        axs = self.gss[key]
        data_dict = self.load_shape_groups()

        for i, (k, data_k) in enumerate(data_dict.items()):
            slv.plot_session_average(data_k, ax=axs[i])
            axs[i].set_title(k)
            gpl.clean_plot(axs[i], i)

    def panel_tasks(self):
        key = "panel_tasks"
        axs = self.gss[key]

        data_dict = self.load_shape_groups()

        for i, (k, data_k) in enumerate(data_dict.items()):
            slv.plot_sampled_stimuli(
                data_k,
                ind=0,
                stim_cat_field="chosen_cat",
                ax=axs[i, 0],
            )
            slv.plot_sampled_stimuli(
                data_k,
                ind=-1,
                stim_cat_field="chosen_cat",
                ax=axs[i, 1],
            )


class SpecificTransitionFigure(SequenceLearningFigure):
    def __init__(
        self,
        info,
        gen,
        var,
        fig_key="specific_transition_figure",
        fwid=3,
        colors=colors,
        exper_data=None,
        **kwargs,
    ):
        self.n_sh = len(np.unique(info[0]))
        fsize = (self.n_sh * 2 * fwid, self.n_sh * fwid + fwid)
        self.info = info
        self.gen = gen
        self.var = var

        cf = u.ConfigParserColor()
        cf.read(config_path)
        params = cf[fig_key]
        self.fig_key = fig_key
        super().__init__(fsize, params, colors=colors, **kwargs)

    def make_gss(self):
        gss = {}

        gen_grid = pu.make_mxn_gridspec(
            self.gs, self.n_sh + 1, self.n_sh, 0, 100, 0, 45, 5, 2
        )
        var_grid = pu.make_mxn_gridspec(
            self.gs, self.n_sh + 1, self.n_sh, 0, 100, 55, 100, 5, 2
        )

        gen_tr_axs = self.get_axs(
            gen_grid[: self.n_sh],
            squeeze=True,
            sharey="all",
        )
        gen_map_ax = self.get_axs(
            gen_grid[-1:, 1],
            squeeze=False,
        )[0, 0]
        gss["panel_gen"] = (gen_tr_axs, gen_map_ax)

        var_tr_axs = self.get_axs(
            var_grid[: self.n_sh],
            squeeze=True,
            sharey="all",
        )
        var_map_ax = self.get_axs(
            var_grid[-1:, 1],
            squeeze=False,
        )[0, 0]
        gss["panel_var"] = (var_tr_axs, var_map_ax)

        self.gss = gss

    def _plot_tr_map(self, quant, key, **kwargs):
        tr_axs, map_ax = self.gss[key]
        slv.plot_decoder_autocorrelation_full(*self.info, quant, axs=tr_axs, **kwargs)
        slv.plot_decoder_autocorrelation_map(*self.info, quant, ax=map_ax, **kwargs)

    def panel_gen(self):
        key = "panel_gen"
        self._plot_tr_map(self.gen, key)

    def panel_var(self):
        key = "panel_var"
        self._plot_tr_map(
            self.var, key, normalize=True, chance=None, y_label="shared variance"
        )


class ShapeSpaceSummary(SequenceLearningFigure):
    def __init__(
        self,
        shape_string,
        fig_key="dec_figure",
        colors=colors,
        exper_data=None,
        **kwargs,
    ):
        fsize = (16, 12)

        cf = u.ConfigParserColor()
        cf.read(config_path)
        params = cf[fig_key]
        self.fig_key = fig_key
        self.shape = shape_string
        self.panel_keys = ("panel_decoding",)
        if exper_data is not None:
            add_data = {"exper_data": exper_data}
            data = kwargs.get("data", {})
            data.update(add_data)
            kwargs["data"] = data
        super().__init__(fsize, params, colors=colors, **kwargs)

    def make_gss(self):
        data = self.load_shape_data(self.shape)
        days = np.unique(data["day"])
        n_pts = len(days)

        n_plot = int(np.ceil(np.sqrt(n_pts)))

        gss = {}
        task_grid = pu.make_mxn_gridspec(self.gs, n_plot, n_plot, 0, 80, 0, 60, 2, 2)
        task_ax = self.get_axs(
            task_grid,
            squeeze=True,
            sharex="all",
            sharey="all",
        )
        gss["panel_bhv"] = task_ax.flatten()

        dec_grid = pu.make_mxn_gridspec(
            self.gs,
            5,
            2,
            0,
            80,
            70,
            100,
            5,
            4,
        )
        dec_axs = self.get_axs(
            dec_grid,
            squeeze=True,
            sharex="all",
            sharey="all",
        )

        dec_change_grid = pu.make_mxn_gridspec(self.gs, 1, 3, 85, 100, 60, 100, 5, 4)
        dec_change_axs = self.get_axs(dec_change_grid, squeeze=True)
        gss["panel_decoding"] = dec_axs
        gss["panel_change_decoding"] = dec_change_axs

        dp_grid = pu.make_mxn_gridspec(self.gs, 1, 3, 85, 100, 0, 40, 5, 4)
        dprime_axs = self.get_axs(dp_grid, squeeze=True)
        gss["panel_change_dprime"] = dprime_axs

        self.gss = gss

    def panel_bhv(self):
        key = "panel_bhv"
        axs = self.gss[key]
        data = self.load_shape_data(self.shape)
        days, inds = np.unique(data["day"], return_index=True)

        r1c, r2c = self.resp_colors
        for i, ind in enumerate(inds):
            slv.plot_sampled_stimuli(
                data,
                ind=ind,
                ax=axs[i],
                color1=r1c,
                color2=r2c,
                stim_cat_field="chosen_cat",
            )

    def panel_decoding(self, **kwargs):
        key = "panel_decoding"
        func_and_name = {
            "category": sla.decode_category,
            "non-category": sla.decode_ortho_category,
            "category\ngeneralization": sla.generalize_category,
            "non-category\ngeneralization": sla.generalize_ortho_category,
            "XOR": sla.decode_xor,
        }
        plot_gen = {
            "generalization": True,
            "non-category generalization": True,
        }
        chance = 0.5
        self._generic_decoding(
            key, func_and_name, chance=chance, plot_gen=plot_gen, **kwargs
        )

    def panel_change_decoding(self):
        key_dec = "panel_decoding"
        key = "panel_change_decoding"
        if self.data.get(key_dec) is None:
            self.panel_decoding()
        dec_dict = self.data[key_dec]
        axs = self.gss[key]
        for i, (region, res_dict) in enumerate(dec_dict["category"].items()):
            cat_dict, xs = res_dict[0]
            days, corr, metrics = [], [], []
            for k, out_k in cat_dict.items():
                days.append(k[0])
                corr.append(np.nanmean(out_k[-2]))
                metrics.append(np.nanmean(out_k[0], axis=0))
            slv.plot_session_change_scatter(
                xs,
                metrics,
                corr,
                None,
                mask_var=False,
                ax=axs[i],
                cm=self.cm_dict[region],
            )

    def panel_change_dprime(self):
        key = "panel_change_dprime"
        axs = self.gss[key]
        if self.data.get(key) is None:
            data = self.load_shape_data()
            regions = self.params.getlist("use_regions")
            winsize = self.params.getint("winsize")
            tbeg = self.params.getint("tbeg")
            tend = self.params.getint("tend")
            step = self.params.getint("winstep")
            uniform_resample = self.params.getboolean("uniform_resample")

            out_dict = {}
            for r in regions:
                out_dict[r] = sla.compute_unit_dprime(
                    data,
                    winsize,
                    tbeg,
                    tend,
                    step,
                    regions=(r,),
                    uniform_resample=uniform_resample,
                )
            self.data[key] = out_dict
        out_dict = self.data[key]
        for i, (region, (xs, dprime, corr, days)) in enumerate(out_dict.items()):
            slv.plot_session_change_scatter(
                xs,
                dprime,
                corr,
                None,
                mask_var=False,
                ax=axs[i],
                cm=self.cm_dict[region],
            )


class ContinuousDecodingFigure(SequenceLearningFigure):
    def __init__(self, fig_key="dec_figure", colors=colors, **kwargs):
        fsize = (4.5, 8)

        cf = u.ConfigParserColor()
        cf.read(config_path)
        params = cf[fig_key]
        self.fig_key = fig_key
        self.panel_keys = ("panel_decoding",)
        super().__init__(fsize, params, colors=colors, **kwargs)

    def make_gss(self):
        gss = {}
        dec_grid = pu.make_mxn_gridspec(self.gs, 4, 2, 0, 100, 0, 100, 10, 5)
        dec_main_ax = self.get_axs(
            dec_grid,
            squeeze=True,
            sharex="all",
            sharey="all",
        )
        dec_inset_axs = np.zeros_like(dec_main_ax)
        bounds = (0.5, 0.5, 0.5, 0.5)
        for i, j in u.make_array_ind_iterator(dec_main_ax.shape):
            ax = dec_main_ax[i, j]
            ax_ins = ax.inset_axes(bounds)
            dec_inset_axs[i, j] = ax_ins
        gss["panel_decoding"] = (dec_main_ax, dec_inset_axs)

        self.gss = gss

    def panel_decoding(self, recompute=True, **kwargs):
        key = "panel_decoding"

        func_and_name = {
            "category": sla.decode_cat_feature,
            "non-category": sla.decode_anticat_feature,
            "category\ngeneralization": sla.generalize_cat_feature,
            "non-category\ngeneralization": sla.generalize_anticat_feature,
        }
        plot_gen = {
            "generalization": True,
            "non-category generalization": True,
        }
        chance = 0
        self._generic_decoding(
            key,
            func_and_name,
            plot_gen,
            chance=chance,
            recompute=recompute,
            inset=True,
            **kwargs,
        )


class DecodingFigure(SequenceLearningFigure):
    def __init__(self, fig_key="dec_figure", colors=colors, **kwargs):
        fsize = (3, 6)

        cf = u.ConfigParserColor()
        cf.read(config_path)
        params = cf[fig_key]
        self.fig_key = fig_key
        self.panel_keys = ("panel_decoding",)
        super().__init__(fsize, params, colors=colors, **kwargs)

    def make_gss(self):
        gss = {}
        dec_grid = pu.make_mxn_gridspec(self.gs, 5, 2, 0, 100, 0, 100, 10, 5)

        gss["panel_decoding"] = self.get_axs(
            dec_grid,
            squeeze=True,
            sharex="all",
            sharey="all",
        )

        self.gss = gss

    def panel_decoding(self, recompute=True, **kwargs):
        key = "panel_decoding"
        func_and_name = {
            "category": sla.decode_category,
            "non-category": sla.decode_ortho_category,
            "category\ngeneralization": sla.generalize_category,
            "non-category\ngeneralization": sla.generalize_ortho_category,
            "XOR": sla.decode_xor,
        }
        plot_gen = {
            "generalization": True,
            "non-category generalization": True,
        }
        chance = 0.5
        self._generic_decoding(
            key,
            func_and_name,
            plot_gen,
            chance=chance,
            recompute=recompute,
            **kwargs,
        )
