import numpy as np
from scipy.spatial import cKDTree as KDTree
from scipy.optimize import minimize
from itertools import combinations_with_replacement
import copy
from ase.neighborlist import NeighborList
import multiprocessing as mp

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

def sigmoid_deriv( x ):
    return np.exp(-x)/( (1.0+np.exp(-x))**2 )

class Neuron(object):
    def __init__(self,n_inputs):
        self.weights = np.ones(n_inputs)

    def evaluate( self, inputs ):
        return sigmoid( self.weights.dot(inputs) )

class OutputNeuron(Neuron):
    def __init__(self,n_inputs):
        Neuron.__init__(self,n_inputs)

    def evaluate( self, inputs ):
        return self.weights.dot(inputs)

class NNPotential(object):
    def __init__( self,pairs=[], n_sym_funcs_per_pair=20, sym_func_width=1.5, Rcut=6.0, Rmin=1.0,n_hidden=30 ):
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

        self.cutoff_func = CutoffFunction( 0.9*Rcut, Rcut )

        n_input_nodes = len(self.pairs)*n_sym_funcs_per_pair
        self.hidden_neurons = []
        for _ in range(n_hidden):
            self.hidden_neurons.append( Neuron(n_input_nodes) )

        self.output_neuron = OutputNeuron(n_hidden)

        # Numpyify the potential
        self.W = np.ones((n_hidden,n_input_nodes))
        self.output_weights = np.ones(n_hidden)

    def find_pairs( self, atoms ):
        elements = []
        for atom in atoms:
            if ( not atom.symbol in elements ):
                elements.append(atom.symbol)
        pair_keys = []
        for pair in combinations_with_replacement(elements,2):
            pair_keys.append(self.create_pair_key(pair))
        return pair_keys

    def create_pair_key( self, pair ):
        sorted_pair = list(pair)
        sorted_pair.sort()
        return "{}-{}".format(sorted_pair[0],sorted_pair[1])

    def total_number_of_weights( self ):
        return len(self.pairs)*self.n_sym_funcs_per_pair*len(self.hidden_neurons) + len(self.hidden_neurons)

    def get_weights( self ):
        weights = []
        for i in range(self.W.shape[0]):
            weights += list(self.W[i,:])
        weights += list(self.output_weights)
        return weights

    def set_weights( self, weights ):
        if ( len(weights) != self.total_number_of_weights() ):
            ng = len(weights)
            nr = self.total_number_of_weights()
            raise ValueError( "The number of weights does not match the required number. Given: {}. Required: {}".format(ng,nr) )

        current = 0
        for i in range(self.W.shape[0]):
            self.W[i,:] = weights[current:current+self.W.shape[1]]
            current += self.W.shape[1]
        self.output_weights[:] = weights[current:]

    def grad_energy_single_atom_wrt_weights( self, inputs ):
        """
        Computes the gradient of the energy of a single atom with respect to the weights
        """
        grad = np.zeros( self.total_number_of_weights() )
        x = self.W.dot(inputs)
        current = 0
        for i in range(self.W.shape[0]):
            grad[current:current+self.W.shape[1]] = self.output_weights[i]*sigmoid_deriv( x[i] )*inputs
            current += self.W.shape[1]

        # Append the gradient with respect to the output weights
        grad[current:] = sigmoid(x)
        return grad

    def grad_total_energy_wrt_weights( self, atoms, nlist ):
        grad = None
        for i in range(len(atoms)):
            inputs = self.get_inputs( atoms, i, nlist )
            if ( grad is None ):
                grad = self.grad_energy_single_atom_wrt_weights(inputs)
            else:
                grad += self.grad_energy_single_atom_wrt_weights(inputs)
        return grad/len(atoms)

    def get_neighbor_list( self, atoms ):
        """
        Returns the neighbor list required to evaluate the potential
        """
        cutoffs = [self.Rcut/2.0 for _ in range(len(atoms))]
        nlist = NeighborList(cutoffs,bothways=True,self_interaction=False)
        nlist.update(atoms)
        return nlist

    def get_inputs( self, atoms, indx, nlist ):
        n_input_nodes = len(self.pairs)*self.n_sym_funcs_per_pair
        inputs = np.zeros(n_input_nodes)
        indices, offsets = nlist.get_neighbors(indx)
        for k,i in enumerate(indices):
            dist = np.sqrt( np.sum(offsets[k]**2) )
            pair = self.create_pair_key( (atoms[indx].symbol,atoms[i].symbol) )
            sym_funcs = self.sym_funcs[pair]
            for sym_func in sym_funcs:
                inputs[sym_func.uid] += sym_func(dist)*self.cutoff_func(dist)
        return inputs

    def get_n_hidden_neurons( self ):
        return self.W.shape[0]

    def get_n_input_nodes( self ):
        return self.W.shape[1]

    def get_potential_energy( self, atoms, indx, nlist ):
        """
        Return the potential energy atom indx
        """
        inputs = self.get_inputs(atoms,indx,nlist)
        neuron_output = sigmoid( self.W.dot(inputs) )
        return self.output_weights.dot(neuron_output)

    def get_force_component( self, atoms, indx, delta ):
        """
        Compute the force on an atom (by central differences)
        """
        orig_positions = atoms.get_positions()
        new_pos = copy.deepcopy(orig_positions)
        new_pos[indx,:] += 0.5*delta
        atoms.set_positions(new_pos)
        energy1 = self.get_potential_energy(atoms,indx)
        new_pos[indx,:] -= delta
        atoms.set_positions(new_pos)
        energy2 = self.get_potential_energy(atoms,indx)
        atoms.set_positions(orig_positions)
        step_size = np.sqrt( np.sum(delta**2) )
        force = (energy1-energy2)/step_size
        return force

    def get_force_vector( self, atoms, indx, step_size=0.04 ):
        delta = np.zeros(3)
        delta[0] = step_size
        fx = self.get_force_component(atoms,indx,delta)
        delta[0] = 0.0
        delta[1] = step_size
        fy = self.get_force_component(atoms,indx,delta)
        delta[1] = 0.0
        delta[2] = step_size
        fz = self.get_force_component(atoms,indx,delta)
        return [fx,fy,fz]

    def get_forces( self, atoms, step_size=0.04 ):
        """
        Compute the forces on all atoms in the structure
        """
        forces = np.zeros((len(atoms),3))
        for i in range(len(atoms)):
            forces[i,:] = self.get_force_vector( atoms,i, step_size )
        return forces

    def get_total_energy( self, atoms, nlist=None ):
        """
        Computes the total energy of the system
        """
        if ( nlist is None ):
            nlist = self.get_neighbor_list(atoms)
        tot_energy = 0.0
        for i in range(len(atoms)):
            tot_energy += self.get_potential_energy( atoms, i, nlist )
        return tot_energy

def get_potential_energy_tuple( args ):
    return args[0].get_potential_energy( args[1], args[2], args[3] )

class NetworkTrainer( object ):
    """
    Class for training the network
    """
    def __init__( self, structures, network, lamb=1.0, force_step_size=0.04, fit_energy=True, fit_forces=True ):
        self.fit_energy = fit_energy
        self.fit_forces = fit_forces
        if ( self.fit_energy==False and self.fit_forces==False ):
            raise ValueError( "At least one of fit_energy and fit_force has to be True" )
        self.structures = structures
        self.network = network
        self.lamb = lamb
        self.force_step_size = force_step_size
        print ("Building neighbor lists")
        self.nlists = [self.network.get_neighbor_list(struct) for struct in self.structures]
        print ("Neigborlists finished")

    def find_reasonable_initial_weights( self ):
        """
        Finds initial weights such that the sigmoid functions simply return 0.5
        """
        inputs = self.network.get_inputs(self.structures[0],0,self.nlists[0])
        for i in range(1,len(self.structures[0])):
            inputs += self.network.get_inputs(self.structures[0],i,self.nlists[0])
        mean_inputs = inputs/len(self.structures[0])
        n_weights = self.network.total_number_of_weights()
        #mean_input_value = np.mean(mean_inputs)
        weights = self.network.get_weights()
        N = len(mean_inputs)
        for i in range(0,len(self.network.hidden_neurons)):
            weights[i*N:i*N+N] = 1.0/(mean_inputs*len(mean_inputs))
        self.network.set_weights(weights)

    def cost_function( self, weights ):
        dE = 0.0
        dFx = 0.0
        dFy = 0.0
        dFz = 0.0
        #print (weights)
        self.network.set_weights(weights)
        grad_E = np.zeros(len(weights))
        for i,atoms in enumerate(self.structures):
            ref_forces = atoms.get_forces()
            ref_energy = atoms.get_potential_energy()
            mean_force = np.mean( np.abs(ref_forces),axis=0 )

            if ( self.fit_energy ):
                tot_energy_nn = self.network.get_total_energy( atoms, nlist=self.nlists[i] )
                diff = ( (tot_energy_nn-ref_energy)/ref_energy )**2
                grad_E += self.grad_cost_func_energy_part_single_structure( tot_energy_nn, ref_energy, i )
                dE += diff
            if ( self.fit_forces ):
                forces_nn = self.network.get_forces( atoms, step_size=self.force_step_size )
                dFx += np.mean( (forces_nn[:,0]-ref_forces[:,0])**2 )/mean_force[0]**2
                dFy += np.mean( (forces_nn[:,1]-ref_forces[:,1])**2 )/mean_force[1]**2
                dFz += np.mean( (forces_nn[:,2]-ref_forces[:,2])**2 )/mean_force[2]**2

        avg_energy_diff = np.sqrt(dE)/len(self.structures)
        cost = (dE + (1.0/3.0)*( dFx + dFy + dFz ))/len(self.structures)
        grad = grad_E + 2.0*self.lamb*weights/len(weights)**2
        print (grad)
        print ("Energy difference: {}".format(avg_energy_diff))
        return cost + self.lamb*self.penalization(), grad

    def grad_cost_func_energy_part_single_structure( self, tot_energy_nn, E_ref, struct_indx ):
        grad = self.network.grad_total_energy_wrt_weights( self.structures[struct_indx], self.nlists[struct_indx] )/E_ref
        return 2.0*(tot_energy_nn-E_ref)*grad/E_ref

    def penalization( self ):
        """
        Returns the penalization term
        """
        weights = np.array( self.network.get_weights() )
        return np.sum(weights**2)/len(weights)**2

    def train( self, method="BFGS", outfile="nnweights.csv", print_msg=True ):
        """
        Training the network
        """
        options = {
            "disp":print_msg,
            "eps":0.5
        }
        self.find_reasonable_initial_weights()
        x0 = self.network.get_weights()
        res = minimize( self.cost_function, x0, method=method, jac=True, options=options, tol=1E-3 )
        np.savetxt(outfile, res["x"], delimiter=",")
        print ( "Neural network weights written to %s"%(outfile) )
