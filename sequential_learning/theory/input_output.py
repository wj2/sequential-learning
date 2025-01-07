import numpy as np

import disentangled.data_generation as ddg
import general.tasks.classification as gtc
import general.utility as u
import general.neural_analysis as na


class DiscreteMixedContext:
    def __init__(
        self,
        *args,
        context_dims=0,
        exhaustive=True,
        **kwargs,
    ):
        if exhaustive:
            self.gen = ddg.MixedDiscreteDataGenerator(*args, **kwargs)
        else:
            self.gen = ddg.NonexhaustiveMixedDiscreteDataGenerator(*args, **kwargs)
        if not u.check_list(context_dims):
            context_dims = np.arange(-context_dims, 0)
        self.context_dims = context_dims
        self.con_set = np.identity(len(context_dims))
        self.rng = np.random.default_rng()

    def sample_stim(self, n_samps=1000, contexts=None):
        stim = self.gen.sample_stim(n_samps)
        if contexts is None:
            contexts = self.context_dims
        use_cons = self.con_set[contexts]
        if len(use_cons.shape) == 1:
            use_cons = np.expand_dims(use_cons, 0)
        if len(use_cons) > 0:
            sampled_cons = self.rng.choice(use_cons, size=n_samps)
            stim[:, self.context_dims] = sampled_cons
        return stim

    def get_all_stim(self):
        return self.gen.get_all_stim(con_dims=self.context_dims)

    def sample_reps(self, n_samps=1000, contexts=None, add_noise=False):
        stim = self.sample_stim(n_samps, contexts=contexts)
        reps = self.gen.get_representation(stim, noise=add_noise)
        return stim, reps


def make_task_set(n_tasks, rel_dims, con_dims, share_tasks=None, share_task_dims=True):
    if share_tasks is None:
        share_tasks = ()
    if not u.check_list(rel_dims):
        rel_dims = np.arange(rel_dims)
    if not u.check_list(con_dims):
        con_dims = np.arange(-con_dims, 0)
    task_sets = []
    if not share_task_dims and len(con_dims) > len(rel_dims):
        raise IOError(
            "there must be at least as many relevant dimensions as contexts when "
            "dimensions are not shared across contexts (rel: {}, con: {}))".format(
                len(rel_dims), len(con_dims)
            )
        )
    for i, cd in enumerate(con_dims):
        if share_task_dims:
            use_dims = rel_dims
        else:
            use_dims = rel_dims[i : i + 1]
        task_sets.append(gtc.LinearTask.make_task_group(n_tasks, use_dims))
    for ind1, ind2 in share_tasks:
        task_sets[ind2] = task_sets[ind1]
    con_tasks = gtc.ContextualTask(*task_sets, c_inds=con_dims)
    return con_tasks


class StaticRepSampler:
    def __init__(
        self,
        xy_seq,
        sigma=0.1,
    ):
        self.n_contexts = len(xy_seq)
        self.xy_seq = xy_seq
        self.rng = np.random.default_rng()
        self.pipe = None

    def make_x_preprocessor(self, **kwargs):
        pipe = na.make_model_pipeline(**kwargs)
        xs = np.concatenate(list(xy[0] for xy in self.xy_seq))
        xs = np.reshape(xs, (xs.shape[0], -1))

        pipe.fit(xs)
        self.pipe = pipe

    def sample_xy_pairs(self, n_samps=1000, contexts=None, add_noise=False, **kwargs):
        if contexts is None:
            contexts = range(self.n_contexts)
        x_all = np.concatenate(list(self.xy_seq[i][0] for i in contexts), axis=0)
        y_all = np.concatenate(list(self.xy_seq[i][1] for i in contexts), axis=0)
        inds = self.rng.choice(x_all.shape[0], n_samps)
        x_samples = x_all[inds]
        y_samples = y_all[inds]
        if len(y_samples.shape) == 1:
            y_samples = np.expand_dims(y_samples, 1)
        if len(x_samples.shape) > 2:
            x_samples = np.reshape(x_samples, (x_samples.shape[0], -1))
        if self.pipe is not None:
            x_samples = self.pipe.transform(x_samples)
        if add_noise:
            x_samples = x_samples + self.sigma * self.rng.normal(
                0, 1, size=x_samples.shape
            )
        return x_samples, y_samples

    def generator(self, batch_size=100, max_samples=10**8, **kwargs):
        for i in range(max_samples):
            yield self.sample_xy_pairs(batch_size, **kwargs)


class InputOutputSampler:
    def __init__(
        self,
        n_tasks,
        n_dims,
        rel_dims,
        con_dims,
        share_tasks=None,
        share_task_dims=True,
        **ddg_kwargs,
    ):
        self.gen = DiscreteMixedContext(n_dims, context_dims=con_dims, **ddg_kwargs)
        self.tasks = make_task_set(
            n_tasks,
            rel_dims,
            con_dims,
            share_tasks=share_tasks,
            share_task_dims=share_task_dims,
        )
        self.n_contexts = len(con_dims) if u.check_list(con_dims) else con_dims

    def sample_xy_pairs(self, n_samps=1000, contexts=None, add_noise=False):
        stim, X = self.gen.sample_reps(
            n_samps=n_samps, contexts=contexts, add_noise=add_noise
        )

        y = self.tasks(stim)
        return X, y

    def get_all_pairs(self):
        stim, reps = self.gen.get_all_stim()
        targ = self.tasks(stim)
        return stim, reps, targ

    def generator(self, batch_size=100, max_samples=10**8, **kwargs):
        for i in range(max_samples):
            yield self.sample_xy_pairs(batch_size, **kwargs)
