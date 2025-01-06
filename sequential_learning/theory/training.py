import numpy as np
import matplotlib.pyplot as plt

import sequential_learning.theory.input_output as slio
import general.torch.feedforward as gtf
import general.plotting as gpl


def _binary_match(y_est, y, cent=0.5):
    y_bin = y_est > cent
    return y_bin == y


class TaskPerformanceTracker:
    def __init__(self, label, X, y, func=_binary_match):
        self.label = label
        self.X = X
        self.y = y
        self.performance = []
        self.func = func

    def on_step(self, net):
        y_est = net.get_output(self.X).detach().numpy()
        self.performance.append(self.func(y_est, self.y))

    def finish(self):
        perf = np.array(self.performance)
        self.performance = []
        return perf


def train_model(
    n_tasks,
    rel_dims,
    con_dims,
    irrel_dims=0,
    hidden_units=100,
    mix_strength=0,
    batch_size=10,
    num_steps=500,
    track_samples=1000,
    add_noise=True,
    sigma=0.1,
    share_tasks=None,
    train_sequence=None,
    share_task_dims=False,
):
    inp_dims = rel_dims + con_dims + irrel_dims
    sampler = slio.InputOutputSampler(
        n_tasks,
        inp_dims,
        rel_dims,
        con_dims,
        mix_strength=mix_strength,
        sigma=sigma,
        share_tasks=share_tasks,
        share_task_dims=share_task_dims,
    )
    
    inp, targs = sampler.sample_xy_pairs(2)
    net = gtf.FeedForwardNetwork(inp.shape[1], (hidden_units,), targs.shape[1])

    loss_tracker = np.zeros((con_dims, num_steps))
    gen_tracker = np.zeros((con_dims, con_dims, num_steps, track_samples, n_tasks))
    trackers = [
        TaskPerformanceTracker(
            j,
            *sampler.sample_xy_pairs(
                track_samples,
                contexts=(j,),
                add_noise=add_noise,
            ),
        )
        for j in range(con_dims)
    ]

    if train_sequence is None:
        train_sequence = range(con_dims)
    for i, ci in enumerate(train_sequence):
        out = net.fit_generator(
            sampler.generator(batch_size, contexts=(ci,), add_noise=add_noise),
            num_steps=num_steps,
            trackers=trackers,
        )
        loss_tracker[i] = out["loss"]
        for j in range(con_dims):
            gen_tracker[i, j] = out["trackers"][j]
    out_dict = {
        "loss": loss_tracker,
        "generalization": gen_tracker,
        "network": net,
        "sampler": sampler,
    }
    return out_dict


def visualize_training(
    out_perf, fwid=3, axs=None, colors=None, cm="Blues", color_bounds=(0.3, 0.9),
):
    if colors is None:
        colors = plt.get_cmap(cm)(np.linspace(*color_bounds, out_perf.shape[0]))
    if axs is None:
        n_plots = len(out_perf)
        f, axs = plt.subplots(1, n_plots, figsize=(fwid * n_plots, fwid), sharey=True)
    for i, perf_i in enumerate(out_perf):
        ax = axs[i]
        xs = np.arange(perf_i.shape[1])
        mu_i = np.mean(perf_i, axis=2)
        for j, mu_ij in enumerate(mu_i):
            if i == j:
                ls = "solid"
            else:
                ls = "dashed"
            ax.plot(xs, mu_ij, color=colors[j], ls=ls)
        gpl.clean_plot(ax, i)
        gpl.add_hlines(0.5, ax)
        ax.set_xlabel("training step")
        if i == 0:
            ax.set_ylabel("fraction correct")
