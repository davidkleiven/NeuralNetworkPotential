import numpy as np
from scipy.spatial import cKDTree as KDTree

class SymmetryFunction(object):
    def __init__( self, sigma, center, uid ):
        self.sigma = sigma
        self.center = center
        self.uid = uid

    def __call__( self, pair_distance ):
        """
        Gaussian symmetry function
        """
        return np.exp( -((pair_distance-self.center)**2)/(2.0*self.sigma**2) )

class CutoffFunction(object):
    def __init__( self, Rmin, Rmax ):
        if ( Rmin >= Rmax ):
            raise ValueError( "Rmin has to be strictly smaller than Rmax" )
        self.Rmin = Rmin
        self.Rmax = Rmax

    def __call__( self, pair_distance ):
        if ( pair_distance < self.Rmin ):
            return 1.0
        elif ( pair_distance > self.Rmax ):
            return 0.0
        else:
            return 0.5*( np.cos(np.pi*(pair_distance-self.Rmin)/(self.Rmax-self.Rmin) ) + 1.0 )

def sigmoid( x ):
    return 1.0/(1.0+np.exp(-x)) - 0.5

class Neuron(object):
    def __init__(self,n_inputs):
        self.weights = np.zeros(n_inputs)

    def evaluate( self, inputs ):
        return sigmoid( self.weights.dot(inputs) )

class OutputNeuron(Neuron):
    def __init__(self,n_inputs):
        Neuron.__init__(self,n_intputs)

    def evaluate( self, inputs ):
        return self.weights.dot(inputs)

class NNPotential(object):
    def __init__( self, atoms, pairs=[], n_sym_funcs_per_pair=20, sym_func_width=1.5, Rcut=6.0, Rmin=1.0,n_hidden=30 ):
        self.layers = layers
        self.pairs = pairs
        self.sym_funcs = {pair:[] for pair in self.pairs}
        self.n_sym_funcs_per_pair = n_sym_funcs_per_pair
        self.Rcut = Rcut
        centers = np.linspace( Rmin, Rcut, n_sym_funcs_per_pair )
        uid = 0
        for pair in self.pairs:
            for mu in centers:
                self.sym_funcs[pair].append( SymmetryFunction(sym_func_width,mu,uid) )
                uid += 1

        self.cutoff_func = CutoffFunction( self, 0.9*Rcut, Rcut )

        n_input_nodes = len(self.pairs)*n_sym_funcs_per_pair
        self.hidden_neurons = []
        for _ in range(n_input_nodes):
            self.hidden_neurons.append( Neuron(n_input_nodes) )

        self.output_neuron = OutputNeuron(n_hidden)

    def get_potential_energy( self, atoms, indx ):
        """
        Return the potential energy atom indx
        """
        n_input_nodes = np.zeros( len(self.pairs)*self.n_sym_funcs_per_pair )
        inputs = np.zeros()
        for i in range(len(atoms)):
            if ( i == atom[indx] ):
                continue
            dist = atom.get_distance(i,indx,mic=True)
            if ( dist > self.Rcut ):
                continue
            pair = "{}-{}".format(atoms[indx].symbol,atoms[i].symbol)
            sym_funcs = self.sym_funcs[pair]
            for sym_func in sym_funcs:
                inputs[sym_func.uid] += sym_func(dist)*self.cutoff_func(dist)

        neuron_output = []
        for neuron in self.hidden_neurons:
            neuron_output.append( neuron.evaluate(inputs) )
        return self.output_neuron.evaluate(neuron_output)
