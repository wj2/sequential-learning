import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sts
import pickle

import general.paper_utilities as pu
import general.utility as u
import general.plotting as gpl
import sequential_learning.auxiliary as slaux
import sequential_learning.analysis as sla
import sequential_learning.visualization as slv
import sequential_learning.ann_representation as slar

config_path = "sequential_learning/sequential_learning/figures.conf"
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

    def only_load_shape_data(
        self, shape="A6", validate=True, with_passive=False, **kwargs
    ):
        data = slaux.load_shape_list(
            (shape,), exclude_invalid=validate, with_passive=with_passive, **kwargs
        )
        if with_passive:
            out = (data[0][shape], data[1][shape])
        else:
            out = data[shape]
        return out

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


class ANNContinualLearning(SequenceLearningFigure):
    def __init__(
        self,
        shapes=("A2", "A3", "A4"),
        fig_key="continual_learning_figure",
        fwid=2,
        **kwargs,
    ):
        cf = u.ConfigParserColor()
        cf.read(config_path)
        params = cf[fig_key]
        self.fig_key = fig_key
        self.shapes = shapes
        fsize = (fwid, fwid * len(shapes))
        super().__init__(fsize, params, colors=colors, **kwargs)

    def make_gss(self):
        gss = {}
        proj_grid = pu.make_mxn_gridspec(
            self.gs,
            1,
            len(self.shapes),
            0,
            100,
            0,
            100,
            1,
            1,
        )
        proj_axs = self.get_axs(proj_grid, squeeze=False, sharex="all", sharey="all")
        gss["panel_learning"] = proj_axs

        self.gss = gss

    def panel_learning(self, retrain=False):
        key = "panel_learning"
        axs = self.gss[key]

        if self.data.get(key) is None or retrain:
            img_reps = slar.get_representations(shapes=self.shapes)


class ANNBoundaryExtrapolationFigure(SequenceLearningFigure):
    def __init__(
        self,
        shape=None,
        exper_data=None,
        fig_key="boundary_extrapolation_figure",
        region="IT",
        fig_folder="",
        uniform_resample=False,
        min_trials=100,
        dec_field="cat_proj",
        gen_field="anticat_proj",
        balance_field=None,
        gen_func=None,
        fwid=2,
        dec_ref=0,
        **kwargs,
    ):
        cf = u.ConfigParserColor()
        cf.read(config_path)
        params = cf[fig_key]
        self.fig_key = fig_key
        self.region = (region,)
        self.fig_folder = fig_folder
        self.gen_func = gen_func
        self.dec_ref = dec_ref
        if exper_data is not None:
            add_data = {"exper_data": exper_data}
            data = kwargs.get("data", {})
            data.update(add_data)
            kwargs["data"] = data
            self.shape = exper_data["shape"].iloc[0]
        elif shape is not None:
            self.shape = shape
        else:
            raise IOError("either data or shape must be provided")

        self.params = params
        self.data = kwargs.pop("data", {})
        n_sessions = self._get_n_sessions()
        fsize = (fwid * 2, fwid * n_sessions)
        super().__init__(fsize, params, data=self.get_data(), colors=colors, **kwargs)

    def _get_n_sessions(self):
        data = self.load_shape_data(shape=self.shape)
        n_sessions = len(np.unique(data["day"]))
        return n_sessions

    def make_gss(self):
        gss = {}
        n_sessions = self._get_n_sessions()
        proj_grid = pu.make_mxn_gridspec(
            self.gs,
            n_sessions,
            2,
            0,
            100,
            0,
            100,
            1,
            1,
        )
        proj_axs = self.get_axs(proj_grid, squeeze=False, sharex="all", sharey="all")
        gss["panel_pattern"] = proj_axs

        self.gss = gss

    def _analysis(self, recompute=False):
        data = self.load_shape_data(shape=self.shape)
        fkey = ("main_analysis", self.shape)
        if self.data.get(fkey) is None or recompute:
            img_reps = slar.get_representations(shapes=(self.shape,))
            dd = {self.shape: data}
            lvs, _, reps = slar.project_lvs(img_reps, dd)[self.shape]
            lv_proj, gen_proj = slar.generalization_pattern(
                lvs, reps, pca=self.params.getfloat("pca")
            )
            _, day_ind = np.unique(data["day"], return_index=True)
            projs = data[["chosen_cat", "cat_proj", "anticat_proj"]]
            projs = list(proj for i, proj in enumerate(projs) if i in day_ind)
            self.data[fkey] = projs, lv_proj, gen_proj
        return self.data[fkey]

    def panel_pattern(self, **kwargs):
        key = "panel_pattern"
        axs = self.gss[key]

        projs, lv_proj, gen_proj = self._analysis(**kwargs)

        feat_dims = (1, 2)
        choice_dim = 0
        for i, proj_i in enumerate(projs):
            feat_i = proj_i.to_numpy()
            slv.plot_gen_map_average(
                feat_i[:, feat_dims],
                feat_i[:, choice_dim],
                ax=axs[i, 0],
                vmin=1,
                vmax=2,
            )
            slv.plot_gen_map_average(lv_proj, gen_proj, ax=axs[i, 1], vmin=0, vmax=1)

    def save_quantification(self, file_=None, use_bf=None, **kwargs):
        if file_ is None:
            file_ = self.fig_key
        if use_bf is None:
            use_bf = self.bf

        projs, lv_proj, gen_proj = self._analysis(**kwargs)
        bhv_feats = list(proj.to_numpy() for proj in projs)
        lv_projs = (lv_proj,) * len(bhv_feats)
        gen_projs = (gen_proj,) * len(bhv_feats)
        quant_out = sla.quantify_error_pattern_sessions(
            lv_projs,
            gen_projs,
            bhv_feats=bhv_feats,
            average_folds=True,
        )

        fname = os.path.join(use_bf, file_)
        pickle.dump(quant_out, open(fname, "wb"))


class PassiveErrorPattern(SequenceLearningFigure):
    def __init__(
        self,
        shape=None,
        exper_data=None,
        fig_key="error_pattern_figure",
        region="IT",
        fig_folder="",
        uniform_resample=False,
        min_trials=100,
        dec_field="cat_proj",
        balance_field="chosen_cat",
        fwid=2,
        dec_ref=0,
        **kwargs,
    ):
        cf = u.ConfigParserColor()
        cf.read(config_path)
        params = cf[fig_key]
        self.fig_key = fig_key
        self.region = (region,)
        self.fig_folder = fig_folder
        self.min_trials = min_trials
        self.dec_field = dec_field
        self.dec_ref = dec_ref
        self.balance_field = balance_field
        if exper_data is not None:
            add_data = {"exper_data": exper_data}
            data = kwargs.get("data", {})
            data.update(add_data)
            kwargs["data"] = data
            self.shape = exper_data["shape"].iloc[0]
        elif shape is not None:
            self.shape = shape
        else:
            raise IOError("either data or shape must be provided")

        self.params = params
        self.data = kwargs.pop("data", {})
        fsize = (fwid * 2, fwid)
        super().__init__(fsize, params, data=self.get_data(), colors=colors, **kwargs)

    def make_gss(self):
        gss = {}
        proj_grid = pu.make_mxn_gridspec(
            self.gs,
            1,
            2,
            0,
            100,
            0,
            100,
            1,
            1,
        )
        proj_axs = self.get_axs(proj_grid, squeeze=False, sharex="all", sharey="all")
        gss["panel_pattern"] = proj_axs

        self.gss = gss

    def _analysis(self, recompute=False):
        data = self.load_shape_data(shape=self.shape)
        fkey = ("main_analysis", self.shape)
        if self.data.get(fkey) is None or recompute:
            t_start = self.params.getfloat("t_start")
            t_end = self.params.getfloat("t_end")
            binsize = self.params.getfloat("binsize")
            binstep = self.params.getfloat("binstep")
            out = sla.error_projection_pattern(
                data,
                self.dec_field,
                regions=self.region,
                t_start=t_start,
                t_end=t_end,
                binsize=binsize,
                binstep=binstep,
                balance_field=self.balance_field,
                uniform_resample=self.uniform_resample,
                min_trials=self.min_trials,
                dec_ref=self.dec_ref,
            )
            self.data[fkey] = out
        return self.data[fkey]

    def panel_pattern(self, **kwargs):
        key = "panel_pattern"
        axs = self.gss[key]

        out_dict = self._analysis(**kwargs)
        projs = out_dict["proj_test"]
        feats = out_dict["feats_test"]
        slv.plot_full_generalization(projs, feats, axs=axs)


class DecoderErrorPatternFigure(SequenceLearningFigure):
    def __init__(
        self,
        shape=None,
        exper_data=None,
        fig_key="error_pattern_figure",
        region="IT",
        fig_folder="",
        uniform_resample=False,
        min_trials=100,
        dec_field="cat_proj",
        balance_field="chosen_cat",
        fwid=2,
        dec_ref=0,
        **kwargs,
    ):
        cf = u.ConfigParserColor()
        cf.read(config_path)
        params = cf[fig_key]
        self.fig_key = fig_key
        self.region = (region,)
        self.fig_folder = fig_folder
        self.uniform_resample = uniform_resample
        self.min_trials = min_trials
        self.dec_field = dec_field
        self.dec_ref = dec_ref
        self.balance_field = balance_field
        if exper_data is not None:
            add_data = {"exper_data": exper_data}
            data = kwargs.get("data", {})
            data.update(add_data)
            kwargs["data"] = data
            self.shape = exper_data["shape"].iloc[0]
        elif shape is not None:
            self.shape = shape
        else:
            raise IOError("either data or shape must be provided")

        self.params = params
        self.data = kwargs.pop("data", {})
        n_sessions = self._get_n_sessions()
        fsize = (fwid * 3, fwid * n_sessions)
        super().__init__(fsize, params, data=self.get_data(), colors=colors, **kwargs)

    def _get_n_sessions(self):
        data = self.load_shape_data(shape=self.shape)
        n_sessions = len(np.unique(data["day"]))
        return n_sessions

    def make_gss(self):
        gss = {}
        n_sessions = self._get_n_sessions()
        proj_grid = pu.make_mxn_gridspec(
            self.gs,
            n_sessions,
            2,
            0,
            100,
            0,
            100,
            1,
            1,
        )
        proj_axs = self.get_axs(proj_grid, squeeze=False, sharex="all", sharey="all")
        gss["panel_pattern"] = proj_axs

        self.gss = gss

    def _analysis(self, recompute=False):
        data = self.load_shape_data(shape=self.shape)
        fkey = ("main_analysis", self.shape)
        if self.data.get(fkey) is None or recompute:
            t_start = self.params.getfloat("t_start")
            binsize = self.params.getfloat("binsize")
            binstep = self.params.getfloat("binstep")
            out = sla.error_projection_pattern(
                data,
                self.dec_field,
                regions=self.region,
                t_start=t_start,
                binsize=binsize,
                binstep=binstep,
                balance_field=self.balance_field,
                uniform_resample=self.uniform_resample,
                min_trials=self.min_trials,
                dec_ref=self.dec_ref,
            )
            self.data[fkey] = out
        return self.data[fkey]

    def panel_pattern(self, **kwargs):
        key = "panel_pattern"
        axs = self.gss[key]

        out_dict = self._analysis(**kwargs)
        projs = out_dict["proj_test"]
        feats = out_dict["feats_test"]
        slv.plot_full_generalization(projs, feats, axs=axs)

    def save_quantification(self, file_=None, use_bf=None, **kwargs):
        if file_ is None:
            file_ = self.fig_key
        if use_bf is None:
            use_bf = self.bf

        out_dict = self._analysis(**kwargs)
        projs = out_dict["proj_test"]
        feats = out_dict["feats_test"]
        quant_out = sla.quantify_error_pattern_sessions(projs, feats)
        fname = os.path.join(use_bf, file_)
        pickle.dump(quant_out, open(fname, "wb"))


class FixationBoundaryExtrapolationFigure(SequenceLearningFigure):
    def __init__(
        self,
        shape=None,
        exper_data=None,
        fig_key="boundary_extrapolation_figure",
        region="IT",
        fig_folder="",
        uniform_resample=False,
        min_trials=100,
        dec_field="cat_proj",
        gen_field="anticat_proj",
        balance_field=None,
        gen_func=slaux.proto_box_mask,
        combine_days_fixation=False,
        fwid=2,
        dec_ref=0,
        **kwargs,
    ):
        cf = u.ConfigParserColor()
        cf.read(config_path)
        params = cf[fig_key]
        self.combine_days_fixation = combine_days_fixation
        self.fig_key = fig_key
        self.region = (region,)
        self.fig_folder = fig_folder
        self.uniform_resample = uniform_resample
        self.min_trials = min_trials
        self.dec_field = dec_field
        self.gen_field = gen_field
        self.gen_func = gen_func
        self.dec_ref = dec_ref
        self.balance_field = balance_field
        if exper_data is not None:
            add_data = {"exper_data": exper_data}
            data = kwargs.get("data", {})
            data.update(add_data)
            kwargs["data"] = data
            self.shape = exper_data["shape"].iloc[0]
        elif shape is not None:
            self.shape = shape
        else:
            raise IOError("either data or shape must be provided")

        self.params = params
        self.data = kwargs.pop("data", {})
        n_sessions = self._get_n_sessions()
        fsize = (fwid * 3, fwid * n_sessions)
        super().__init__(fsize, params, data=self.get_data(), colors=colors, **kwargs)

    def _get_n_sessions(self):
        data, _ = self.load_shape_data(shape=self.shape, with_passive=True)
        n_sessions = len(np.unique(data["day"]))
        return n_sessions

    def make_gss(self):
        gss = {}
        n_sessions = self._get_n_sessions()
        proj_grid = pu.make_mxn_gridspec(
            self.gs,
            n_sessions,
            3,
            0,
            100,
            0,
            100,
            1,
            1,
        )
        proj_axs = self.get_axs(proj_grid, squeeze=False, sharex="all", sharey="all")
        gss["panel_pattern"] = proj_axs

        self.gss = gss

    def _analysis(self, recompute=False):
        data, fix_data = self.load_shape_data(shape=self.shape, with_passive=True)
        fkey = ("main_analysis", self.shape)
        if self.data.get(fkey) is None or recompute:
            t_start = self.params.getfloat("t_start")
            binsize = self.params.getfloat("binsize")
            binstep = self.params.getfloat("binstep")
            active_out = sla.generalize_projection_pattern(
                data,
                self.dec_field,
                gen_func=self.gen_func,
                gen_field=self.gen_field,
                regions=self.region,
                t_start=t_start,
                binsize=binsize,
                binstep=binstep,
                balance_field=self.balance_field,
                uniform_resample=self.uniform_resample,
                min_trials=self.min_trials,
                dec_ref=self.dec_ref,
            )
            bhv_feats = data[["chosen_cat", "cat_proj", "anticat_proj"]]
            bhv_feats = list(bff.to_numpy() for bff in bhv_feats)

            fix_out = sla.fixation_generalization_pattern(
                fix_data,
                bhv_days=data["day"],
                gen_func=self.gen_func,
                gen_field=self.gen_field,
                regions=self.region,
                dec_ref=self.dec_ref,
                min_trials=self.min_trials,
                bhv_feats=bhv_feats,
                combine_days=self.combine_days_fixation,
            )
            self.data[fkey] = (active_out, fix_out)
        return self.data[fkey]

    def panel_pattern(self, **kwargs):
        key = "panel_pattern"
        axs = self.gss[key]

        smooth_radius = self.params.getfloat("smooth_radius")

        out_dict_active, out_dict_fix = self._analysis(**kwargs)
        feats_bhv = out_dict_fix["bhv_feats"]
        projs = out_dict_fix["proj_gen"]
        feats = out_dict_fix["feats_gen"]

        projs_active = out_dict_active["proj_gen"]
        feats_active = out_dict_active["feats_gen"]
        if self.combine_days_fixation:
            feats_bhv = feats_bhv * len(feats_active)
            projs = projs * len(projs_active)
            feats = feats * len(feats_active)
        slv.plot_fixation_generalization(
            feats_bhv,
            projs,
            feats,
            axs=axs[:, (0, 2)],
            average_folds=True,
            smooth_radius=smooth_radius,
        )

        slv.plot_full_generalization(
            projs_active,
            feats_active,
            axs=axs[:, :2],
            average_folds=True,
            smooth_radius=smooth_radius,
        )

    def save_quantification(self, file_=None, use_bf=None, **kwargs):
        if file_ is None:
            file_ = self.fig_key
        if use_bf is None:
            use_bf = self.bf

        out_active, out_fix = self._analysis(**kwargs)
        projs_active = out_active["proj_gen"]
        feats_active = out_active["feats_gen"]

        projs_fix = out_fix["proj_gen"]
        feats_fix = out_fix["feats_gen"]
        bhv_feats_fix = out_fix["bhv_feats"]

        quant_fix = sla.quantify_error_pattern_sessions(
            projs_fix,
            feats_fix,
            bhv_feats=bhv_feats_fix,
            average_folds=True,
        )
        quant_active = sla.quantify_error_pattern_sessions(
            projs_active,
            feats_active,
            average_folds=True,
        )
        quant_out = {"task": quant_active, "fixation": quant_fix}

        fname = os.path.join(use_bf, file_)
        pickle.dump(quant_out, open(fname, "wb"))


class PrototypeBoundaryExtrapolationFigure(SequenceLearningFigure):
    def __init__(
        self,
        shape=None,
        exper_data=None,
        fig_key="prototype_boundary_extrapolation_figure",
        region="IT",
        fig_folder="",
        uniform_resample=False,
        fwid=2,
        **kwargs,
    ):
        cf = u.ConfigParserColor()
        cf.read(config_path)
        params = cf[fig_key]
        self.fig_key = fig_key
        self.region = (region,)
        self.fig_folder = fig_folder
        self.uniform_resample = uniform_resample
        if exper_data is not None:
            add_data = {"exper_data": exper_data}
            data = kwargs.get("data", {})
            data.update(add_data)
            kwargs["data"] = data
            self.shape = exper_data["shape"].iloc[0]
        elif shape is not None:
            self.shape = shape
        else:
            raise IOError("either data or shape must be provided")

        self.params = params
        self.data = kwargs.pop("data", {})
        n_panels = self._get_n_panels()
        fsize = (fwid * 3, fwid * (n_panels + 1))
        super().__init__(fsize, params, data=self.get_data(), colors=colors, **kwargs)

    def _get_n_panels(self):
        out_res = self._analysis()
        n_panels = len(out_res["proj"])
        return n_panels

    def make_gss(self):
        gss = {}
        n_panels = self._get_n_panels()
        trs = int(np.round(100 / (n_panels + 1)))
        proj_grid = pu.make_mxn_gridspec(
            self.gs,
            n_panels,
            2,
            trs + 2,
            100,
            0,
            100,
            1,
            1,
        )
        proj_axs = self.get_axs(proj_grid, squeeze=False, sharex="all", sharey="all")
        targ_ax = self.get_axs((self.gs[:trs, :],), squeeze=False)[0, 0]
        gss["panel_pattern"] = (proj_axs, targ_ax)

        self.gss = gss

    def _analysis(self, recompute=False):
        data = self.load_shape_data(shape=self.shape)
        fkey = ("main_analysis", self.shape)
        if self.data.get(fkey) is None or recompute:
            tbeg = self.params.getfloat("t_start")
            binsize = self.params.getfloat("binsize")
            out_proto, out_nonproto = sla.prototype_extrapolation_info(
                data, tbeg=tbeg, winsize=binsize, regions=self.region
            )
            out_res = sla.prototype_extrapolation(out_proto, out_nonproto)
            self.data[fkey] = out_res
        return self.data[fkey]

    def panel_pattern(self, **kwargs):
        key = "panel_pattern"
        axs, ax_targ = self.gss[key]

        out_dict = self._analysis(**kwargs)
        projs = out_dict["proj"]
        feats = out_dict["feats_comb"]
        slv.plot_full_generalization(projs, feats, axs=axs, average_folds=True)

        out = sla.quantify_task_error_lr_sessions(projs, feats)
        slv.visualize_task_error_scatter(
            out["score_feat"], out["score"], ax=ax_targ,
        )
        # slv.visualize_task_coeffs(out, ax=ax_targ)



class EarlyPrototypeBoundaryExtrapolationFigure(PrototypeBoundaryExtrapolationFigure):
    def __init__(
        self,
        fig_key="early_prototype_boundary_extrapolation_figure",
        **kwargs,
    ):
        super().__init__(fig_key=fig_key, **kwargs)
        

class LatePrototypeBoundaryExtrapolationFigure(PrototypeBoundaryExtrapolationFigure):
    def __init__(
        self,
        fig_key="late_prototype_boundary_extrapolation_figure",
        **kwargs,
    ):
        super().__init__(fig_key=fig_key, **kwargs)


class BoundaryExtrapolationFigure(SequenceLearningFigure):
    def __init__(
        self,
        shape=None,
        exper_data=None,
        fig_key="boundary_extrapolation_figure",
        region="IT",
        fig_folder="",
        uniform_resample=False,
        min_trials=100,
        dec_field="cat_proj",
        gen_field="anticat_proj",
        balance_field=None,
        gen_func=None,
        fwid=2,
        dec_ref=0,
        **kwargs,
    ):
        cf = u.ConfigParserColor()
        cf.read(config_path)
        params = cf[fig_key]
        self.fig_key = fig_key
        self.region = (region,)
        self.fig_folder = fig_folder
        self.uniform_resample = uniform_resample
        self.min_trials = min_trials
        self.dec_field = dec_field
        self.gen_field = gen_field
        self.gen_func = gen_func
        self.dec_ref = dec_ref
        self.balance_field = balance_field
        if exper_data is not None:
            add_data = {"exper_data": exper_data}
            data = kwargs.get("data", {})
            data.update(add_data)
            kwargs["data"] = data
            self.shape = exper_data["shape"].iloc[0]
        elif shape is not None:
            self.shape = shape
        else:
            raise IOError("either data or shape must be provided")

        self.params = params
        self.data = kwargs.pop("data", {})
        n_sessions = self._get_n_sessions()
        fsize = (fwid * 3, fwid * n_sessions)
        super().__init__(fsize, params, data=self.get_data(), colors=colors, **kwargs)

    def _get_n_sessions(self):
        data = self.load_shape_data(shape=self.shape)
        n_sessions = len(np.unique(data["day"]))
        return n_sessions

    def make_gss(self):
        gss = {}
        n_sessions = self._get_n_sessions()
        proj_grid = pu.make_mxn_gridspec(
            self.gs,
            n_sessions,
            2,
            0,
            100,
            0,
            100,
            1,
            1,
        )
        proj_axs = self.get_axs(proj_grid, squeeze=False, sharex="all", sharey="all")
        gss["panel_pattern"] = proj_axs

        self.gss = gss

    def _analysis(self, recompute=False):
        data = self.load_shape_data(shape=self.shape)
        fkey = ("main_analysis", self.shape)
        if self.data.get(fkey) is None or recompute:
            t_start = self.params.getfloat("t_start")
            binsize = self.params.getfloat("binsize")
            binstep = self.params.getfloat("binstep")
            out = sla.generalize_projection_pattern(
                data,
                self.dec_field,
                gen_field=self.gen_field,
                gen_func=self.gen_func,
                regions=self.region,
                t_start=t_start,
                binsize=binsize,
                binstep=binstep,
                balance_field=self.balance_field,
                uniform_resample=self.uniform_resample,
                min_trials=self.min_trials,
                dec_ref=self.dec_ref,
            )
            self.data[fkey] = out
        return self.data[fkey]

    def panel_pattern(self, **kwargs):
        key = "panel_pattern"
        axs = self.gss[key]

        out_dict = self._analysis(**kwargs)
        projs = out_dict["proj_gen"]
        feats = out_dict["feats_gen"]
        slv.plot_full_generalization(projs, feats, axs=axs, average_folds=True)


class DecisionLearningFigure(SequenceLearningFigure):
    def __init__(
        self,
        pre_shapes=None,
        gen_shape=None,
        fig_key="decision_learning_figure",
        exper_data=None,
        region="IT",
        fig_folder="",
        uniform_resample=False,
        min_trials=100,
        use_fields=("chosen_cat", "cat_proj", "anticat_proj"),
        **kwargs,
    ):
        fsize = (10, 8)
        cf = u.ConfigParserColor()
        cf.read(config_path)
        params = cf[fig_key]
        self.fig_key = fig_key
        self.region = (region,)
        self.fig_folder = fig_folder
        self.uniform_resample = uniform_resample
        self.min_trials = min_trials
        self.use_fields = list(use_fields)
        if exper_data is not None:
            add_data = {"exper_data": exper_data}
            data = kwargs.get("data", {})
            data.update(add_data)
            kwargs["data"] = data
            self.shape_sequence = list(exper_data.keys())
        elif pre_shapes is not None and gen_shape is not None:
            self.shape_sequence = tuple(pre_shapes) + (gen_shape,)
        else:
            raise IOError("either data or shape list must be provided")
        if pre_shapes is None and gen_shape is None:
            self.pre_shapes = self.shape_sequence[:-1]
            self.gen_shape = self.shape_sequence[-1]
        else:
            self.pre_shapes = pre_shapes
            self.gen_shape = gen_shape
        self.pre_key = "-".join(self.pre_shapes)
        self.gen_key = self.gen_shape
        super().__init__(fsize, params, colors=colors, **kwargs)

    def make_gss(self):
        gss = {}

        data_dict = self.load_shape_groups()
        n_sessions = len(np.unique(data_dict[self.gen_key]["day"]))
        n_grid = int(np.ceil(np.sqrt(n_sessions)))
        proj_grid = pu.make_mxn_gridspec(
            self.gs,
            n_grid,
            n_grid,
            0,
            70,
            0,
            100,
            2,
            2,
        )
        proj_axs = self.get_axs(proj_grid, squeeze=True, sharex="all", sharey="all")
        gss["panel_proj"] = proj_axs

        overlap_grid = pu.make_mxn_gridspec(self.gs, 1, 1, 75, 100, 0, 40, 2, 2)
        overlap_axs = self.get_axs(
            overlap_grid,
            sharex="all",
            sharey="all",
        )
        gss["panel_overlap"] = overlap_axs[0, 0]

        self.gss = gss

    def _analysis(self):
        data_dict = self.load_shape_groups()
        fkey = ("main_analysis", self.pre_key, self.gen_key)
        if self.data.get(fkey) is None:
            t_start = self.params.getfloat("t_start")
            t_end = self.params.getfloat("t_end")
            binsize = self.params.getfloat("binsize")
            binstep = self.params.getfloat("binstep")
            out = sla.combined_pregen_decoder(
                data_dict,
                self.pre_shapes,
                self.gen_shape,
                regions=self.region,
                t_start=t_start,
                t_end=t_end,
                binsize=binsize,
                binstep=binstep,
                uniform_resample=self.uniform_resample,
                stim_field=self.use_fields,
                min_trials=self.min_trials,
            )
            self.data[fkey] = out
        return self.data[fkey]

    def panel_proj(self):
        key = "panel_proj"
        axs = self.gss[key].flatten()

        s_pt = self.params.getfloat("pt_size")
        kde_range = self.params.getlist("kde_range", typefunc=float)
        n_pts = self.params.getint("kde_points")
        colormap = plt.get_cmap(self.params.get("colormap"))

        pts = np.linspace(*kde_range, n_pts)
        xs, ys = np.meshgrid(pts, pts)
        xs_flat = np.reshape(xs, (-1,))
        ys_flat = np.reshape(ys, (-1,))
        pts_eval = np.stack((xs_flat, ys_flat), axis=0)

        out_dict = self._analysis()
        dv_ind = 0
        t_ind = 0
        pre_vec = out_dict[self.pre_key]["dvs"][0][..., dv_ind, t_ind]
        post_vec = out_dict[self.gen_key]["dvs"][-1][..., dv_ind, t_ind]
        for i, pop_i in enumerate(out_dict[self.gen_key]["pops"]):
            feat_i = out_dict[self.gen_key]["feats"][i][:, dv_ind]
            pop_i = pop_i[..., t_ind]
            pre_proj = pre_vec @ pop_i
            post_proj = post_vec @ pop_i
            pts_mu = np.mean(np.stack((pre_proj, post_proj), axis=-1), axis=0)
            feat_vals = np.unique(feat_i)
            maps = np.zeros((len(feat_vals), n_pts, n_pts))
            for j, fv in enumerate(feat_vals):
                kde_i = sts.gaussian_kde(pts_mu[feat_i == fv].T)
                maps[j] = np.reshape(kde_i(pts_eval), (n_pts, n_pts))

            map_diff = maps[1] - maps[0]
            extreme = np.max(np.abs(map_diff))
            axs[i].pcolormesh(
                pts, pts, map_diff, vmin=-extreme, vmax=extreme, cmap=colormap
            )
            axs[i].scatter(
                *pts_mu.T,
                c=feat_i,
                s=s_pt,
                alpha=0.1,
                cmap=colormap,
            )
            axs[i].set_aspect("equal")
            gpl.clean_plot(axs[i], i)

    def panel_overlap(self):
        key = "panel_overlap"
        ax = self.gss[key]

        out_dict = self._analysis()
        slv.plot_choice_corr(
            out_dict[self.gen_key]["dvs"],
            ax=ax,
            ref_dv=out_dict[self.pre_key]["dvs"][-1],
        )


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
        min_trials=100,
        use_fields=("cat_proj", "anticat_proj"),
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
        self.min_trials = min_trials
        self.use_fields = list(use_fields)
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

        proj_grids = pu.make_mxn_gridspec(self.gs, 1, 2, 85, 100, 0, 40, 4, 4)
        gss["panel_proj_learning"] = self.get_axs(
            proj_grids, sharey="all", squeeze=True
        )

        self.gss = gss

    def _analysis(
        self,
        recompute=False,
    ):
        data_dict = self.load_shape_groups()
        if self.data.get("main_analysis") is None or recompute:
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
                stim_field=self.use_fields,
                min_trials=self.min_trials,
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
        ang_ax.set_xlabel("task alignment")
        gpl.clean_plot(ang_ax, 1)

    def panel_dec(self, **kwargs):
        key = "panel_dec"
        vis_ax, ang_ax = self.gss[key]
        out = self._analysis(**kwargs)

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
            gpl.clean_3d_plot(ax)
            fp = os.path.join(
                self.fig_folder,
                "vis_{}-{}.mp4".format(*self.shape_sequence),
            )
            gpl.rotate_3d_plot(f, ax, fp, fps=30, n_frames=200, dpi=300)

        v_a2 = u.make_unit_vector(out[s1]["dvs"][-1][..., 0, 0])
        v_a2_i = u.make_unit_vector(out[s1]["dvs"][-2][..., 0, 0])
        v_a3 = u.make_unit_vector(out[s2]["dvs"][0][..., 0, 0])
        v_a3_i = u.make_unit_vector(out[s2]["dvs"][1][..., 0, 0])

        ang_ax.hist((v_a2 @ v_a3.T).flatten(), density=True)
        ang_ax.hist((v_a2_i @ v_a2.T).flatten(), density=True)
        ang_ax.hist((v_a3_i @ v_a3.T).flatten(), density=True)

        ang_ax.set_xlabel("decision vector overlap")
        ang_ax.set_ylabel("density")
        gpl.clean_plot(ang_ax, 0)

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

    def panel_proj_learning(self):
        key = "panel_proj_learning"
        axs = self.gss[key]

        out_data = self._analysis()
        s1, s2 = self.shape_sequence
        test_trls = self.params.getint("proj_test_trls")
        avg_wid = self.params.getint("proj_avg_wid")

        out = sla.choice_projection_tc(
            out_data[s1],
            out_data[s2],
            test_trls=test_trls,
            avg_wid=avg_wid,
        )

        xs_len = len(out["s1 on s1"])
        gpl.plot_colored_line(np.arange(xs_len), out["s1 on s1"], ax=axs[0])
        gpl.plot_colored_line(
            np.arange(xs_len), out["s1 on s2"], ax=axs[0], cmap="Reds"
        )
        gpl.plot_colored_line(
            np.arange(xs_len), out["s1 corr"], ax=axs[0], cmap="Greys"
        )
        gpl.plot_colored_line(np.arange(xs_len), out["s2 on s1"], ax=axs[1])
        gpl.plot_colored_line(
            np.arange(xs_len), out["s2 on s2"], ax=axs[1], cmap="Reds"
        )
        gpl.plot_colored_line(
            np.arange(xs_len), out["s2 corr"], ax=axs[1], cmap="Greys"
        )
        gpl.add_hlines(0.5, axs[0])
        gpl.add_hlines(0.5, axs[1])
        gpl.clean_plot(axs[0], 0)
        gpl.clean_plot(axs[1], 1)


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
