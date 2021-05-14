# Based on OWMeta C. Elegans connectome.
# Parameters from https://arxiv.org/abs/1310.6689
# UPHESS - EPFL
# Authors: Matteo Santoro
# Date: May 2021

# Packages
import numpy as np                                                          # For current directory to read and write files
import argparse                                                             # For options
from tqdm import tqdm                                                                 # For visualizing progress bar

from types import SimpleNamespace

# OWMeta imports
from owmeta_core.command import OWM
from owmeta.worm import Worm
from owmeta.neuron import Neuron
from owmeta_core.context import Context

# ODE
from scipy.integrate import ode
# ****************************************************************************** #
nnum = 302
default_params = {
             'root': '/home/paperspace/motifs/celegans_ode/',
             'exc_path': 'structure/exc.npy',
             'exc_param': SimpleNamespace(**{'base_weight':1, 'rest':0}),
             'inh_path': 'structure/inh.npy',
             'inh_param': SimpleNamespace(**{'base_weight':1, 'rest':0}),
             'gj_path': 'structure/gj.npy',
             'gj_param': SimpleNamespace(**{'base_weight':1}),
             'neuron_param': SimpleNamespace(
                                          **{'beta':1, 'synrise':1, 'syndecay':1,
                                           'conductance':1, 'capacity':1, 'threshold':1,
                                           'rest_potential':1}
                                          ),
             'nnum': nnum,
             'initial_conditions': np.zeros((2*nnum,)),
             'input': lambda t: np.ones((nnum,)),
             'solver_param' : SimpleNamespace(**{'time': 1000,'dt': 1})
             }
# ****************************************************************************** #
# Build the adjacency matrix of the graph of the C. Elegans brain
class ode_solver:
    def __init__(self,
                 root, exc_path, exc_param, inh_path, inh_param, gj_path, gj_param,
                 neuron_param, nnum, initial_conditions, input, solver_param
                ):
        self.root = root
        self.exc_path = exc_path
        self.exc_param = exc_param
        self.inh_path = inh_path
        self.inh_param = inh_param
        self.gj_path = gj_path
        self.gj_param = gj_param
        self.nnum = nnum
        try:
            self.exc_adj = np.load(self.root + self.exc_path)
            self.inh_adj = np.load(self.root + self.inh_path)
            self.gj_adj = np.load(self.root + self.inh_path)
        except:
            self.build_matrix()
        self.neuron_param = neuron_param
        self.initial_conditions = initial_conditions
        self.input = input
        self.solver_param = solver_param

    def _f(self):
        nnum = self.exc_adj.shape[0]
        def sigmoid(x, beta):
            return 1 / (1 + np.exp(-beta*x))
        def f(t,y):
            y1 = y.copy()
            beta_sigmoid = lambda x: sigmoid(x,self.neuron_param.beta)
            y1[nnum:2*nnum] = (self.neuron_param.synrise *
                               np.multiply(
                                   np.array(list(map(beta_sigmoid, y[:nnum] - self.neuron_param.threshold))),
                                   1 - y[nnum:2*nnum]
                                   ) - self.neuron_param.syndecay * y[nnum:2*nnum]
                               )
            gj_matrix = np.multiply(np.expand_dims(y[:nnum], 1) - np.expand_dims(y[:nnum], 0), self.gj_adj)
            exc_matrix = np.multiply(np.expand_dims(y[:nnum], 1) - self.exc_param.rest, self.exc_adj)
            inh_matrix = np.multiply(np.expand_dims(y[:nnum], 1) - self.inh_param.rest, self.inh_adj)
            y1[:nnum] = - self.neuron_param.conductance / self.neuron_param.capacity * (y[:nnum] - self.neuron_param.rest_potential) - np.sum(gj_matrix + exc_matrix + inh_matrix, axis = 1) + self.input(t)
        return f

    def build_solver(self):
        self.solver = ode(self._f())
        self.solver = self.solver.set_integrator('vode')
        self.solver.set_initial_value(self.initial_conditions)

    def solution(self):
        times = []
        solution = []
        with tqdm(total = self.solver_param.time) as pbar:
            while self.solver.successful() and self.solver.t < self.solver_param.time:
                times.append(self.solver.t + self.solver_param.dt)
                solution.append(self.solver.integrate(self.solver.t + self.solver_param.dt))
                pbar.update(self.solver_param.dt)
        return times, solution

    def build_matrix(self):
        conn = OWM().connect()
        ctx = conn(Context)(ident='http://openworm.org/data')
        net = ctx.stored(Worm).query().neuron_network()
        neurons = sorted(
            list(net.neuron()),
            key=lambda x: x.name()
        )
        neuron_names = [n.name() for n in neurons]
        n_neurons = len(neuron_names)
        neuron_ids = range(0, n_neurons)
        name2id = dict([(name, id) for name, id in zip(neuron_names, neuron_ids)])
        neuron_neurotransmitters = [n.neurotransmitter() for n in neurons]

        def is_inhibitory(synapse):
            if synapse.synclass() is not None:
                return 'GABA' in synapse.synclass()
            else:
                return 'GABA' in synapse.pre_cell().neurotransmitter()

        connections = list(net.synapse())
        exc_syn_matrix = np.zeros((n_neurons, n_neurons))
        inh_syn_matrix = np.zeros((n_neurons, n_neurons))
        gj_matrix = np.zeros((n_neurons, n_neurons))
        for connection in connections:
            pre_cell = connection.pre_cell()
            post_cell = connection.post_cell()
            weight = connection.number()
            try:
                if connection.syntype() == 'send': #Chemical synapses
                    if is_inhibitory(connection):
                        inh_syn_matrix[
                            name2id[pre_cell.name()],
                            name2id[post_cell.name()]
                        ] += weight
                    else:
                        exc_syn_matrix[
                            name2id[pre_cell.name()],
                            name2id[post_cell.name()]
                        ] += weight
                elif connection.syntype() == 'gapJunction':
                    gj_matrix[
                        name2id[pre_cell.name()],
                        name2id[post_cell.name()]
                    ] = weight
            except KeyError:
                pass # Non-neuron was found.
        # Save results in npy format
        np.save(self.root + self.exc_path, exc_syn_matrix)
        np.save(self.root + self.inh_path, inh_syn_matrix)
        np.save(self.root + self.gj_path, gj_matrix)

        return exc_syn_matrix, inh_syn_matrix, gj_matrix
