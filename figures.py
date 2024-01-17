
import os
import numpy as np

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

    def load_shape_data(self, shape="A6", max_files=np.inf, validate=True):
        if self.data.get("exper_data") is None:
            data = gio.Dataset.from_readfunc(
                slaux.load_kiani_data_folder,
                os.path.join(self.params.get("data_folder"), shape),
                max_files=max_files,
                sort_by="day",
            )
            if validate:
                data = slaux.filter_valid(data)
            self.data["exper_data"] = data
        return self.data["exper_data"]

    @property
    def color_dict(self):
        regions = self.params.getlist("use_regions")
        color_dict = {
            r: self.params.getcolor("{}_color".format(r)) for r in regions
        }
        return color_dict

    @property
    def resp_colors(self):
        r1 = self.params.getcolor("r1_color")
        r2 = self.params.getcolor("r2_color")
        return r1, r2

    @property
    def cm_dict(self):
        regions = self.params.getlist("use_regions")
        color_dict = {
            r: self.params.get("{}_cm".format(r)) for r in regions
        }
        return color_dict
    
    def _generic_decoding(
            self,
            key,
            func_and_name,
            plot_gen=None,
            chance=.5,
            recompute=False,
            inset=False,
            scatter_bound=(-.75, .75),
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
        if self.data.get(key) is None or recompute:
            data = self.load_shape_data()
            var_ratio = sla.compute_var_ratio(data)
            data_session = data.session_mask(var_ratio > var_thr)
            outs = {}
            for r in regions:
                args = (data_session, winsize, tbeg, tend, step)
                kwargs_stim = {"regions": (r,)}
                kwargs_sacc = {"regions": (r,), "time_zero_field": "fp_off"}
                for k, func in func_and_name.items():
                    out_stim = func(*args, **kwargs_stim)
                    out_sacc = func(*args, **kwargs_sacc)
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
                        ms=.1,
                    )
                    slv.plot_decoding_scatter(
                        *out_ij[r][1],
                        ax=axs_inset[j, 1],
                        cmap=self.cm_dict[r],
                        plot_gen=plot_gen.get(k, False),
                        x_range=scatter_bound,
                        y_range=scatter_bound,
                        ms=.1,
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
        self.panel_keys = (
            "panel_decoding",
        )
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
        task_grid = pu.make_mxn_gridspec(
            self.gs, n_plot, n_plot, 0, 100, 0, 60, 2, 2
        )
        task_ax = self.get_axs(
            task_grid, squeeze=True, sharex="all", sharey="all",
        )
        gss["panel_bhv"] = task_ax.flatten()

        dec_grid = pu.make_mxn_gridspec(
            self.gs, 5, 2, 0, 100, 70, 100, 5, 4,
        )
        dec_axs = self.get_axs(
            dec_grid, squeeze=True, sharex="all", sharey="all",
        )
        gss["panel_decoding"] = dec_axs

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

    def panel_decoding(self):
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
        chance = .5
        self._generic_decoding(key, func_and_name, chance=chance, plot_gen=plot_gen)

                
class ContinuousDecodingFigure(SequenceLearningFigure):
    def __init__(self, fig_key="dec_figure", colors=colors, **kwargs):
        fsize = (4.5, 8)

        cf = u.ConfigParserColor()
        cf.read(config_path)
        params = cf[fig_key]
        self.fig_key = fig_key
        self.panel_keys = (
            "panel_decoding",
        )
        super().__init__(fsize, params, colors=colors, **kwargs)

    def make_gss(self):
        gss = {}
        dec_grid = pu.make_mxn_gridspec(
            self.gs, 4, 2, 0, 100, 0, 100, 10, 5
        )
        dec_main_ax = self.get_axs(
            dec_grid, squeeze=True, sharex="all", sharey="all",
        )
        dec_inset_axs = np.zeros_like(dec_main_ax)
        bounds = (.5, .5, .5, .5)
        for (i, j) in u.make_array_ind_iterator(dec_main_ax.shape):
            ax = dec_main_ax[i, j]
            ax_ins = ax.inset_axes(bounds)
            dec_inset_axs[i, j] = ax_ins
        gss["panel_decoding"] = (dec_main_ax, dec_inset_axs)

        self.gss = gss

    def panel_decoding(self, recompute=True):
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
            key, func_and_name, plot_gen, chance=chance, recompute=recompute, inset=True,
        )


class DecodingFigure(SequenceLearningFigure):
    def __init__(self, fig_key="dec_figure", colors=colors, **kwargs):
        fsize = (3, 6)

        cf = u.ConfigParserColor()
        cf.read(config_path)
        params = cf[fig_key]
        self.fig_key = fig_key
        self.panel_keys = (
            "panel_decoding",
        )
        super().__init__(fsize, params, colors=colors, **kwargs)

    def make_gss(self):
        gss = {}
        dec_grid = pu.make_mxn_gridspec(
            self.gs, 5, 2, 0, 100, 0, 100, 10, 5
        )

        gss["panel_decoding"] = self.get_axs(
            dec_grid, squeeze=True, sharex="all", sharey="all",
        )

        self.gss = gss

    def panel_decoding(self, recompute=True):
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
        chance = .5
        self._generic_decoding(
            key, func_and_name, plot_gen, chance=chance, recompute=recompute
        )
