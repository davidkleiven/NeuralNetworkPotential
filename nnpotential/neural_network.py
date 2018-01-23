from mpi4py import MPI
import numpy as np
from scipy.spatial import cKDTree as KDTree
from scipy.optimize import minimize
from itertools import combinations_with_replacement
import copy
from ase.neighborlist import NeighborList
from matplotlib import pyplot as plt
from ase.visualize import view
import time

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

    def deriv( self, pair_distance ):
        eta = 1.0/(2.0*self.sigma**2)
        return -2.0*eta*(pair_distance-self.center)*self.__call__(pair_distance)

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

    def deriv( self, pair_distance ):
        """
        Compute the gradient of the cutoff function
        """
        if ( pair_distance < self.Rmin or pair_distance > self.Rmax ):
            return 0.0
        return -0.5*np.pi*np.sin( (np.pi*(pair_distance-self.Rmin)/(self.Rmax-self.Rmin) ) )/( self.Rmax-self.Rmin )

def sigmoid( x ):
    return 1.0/(1.0+np.exp(-x)) - 0.5

def sigmoid_deriv( x ):
    return np.exp(-x)/( (1.0+np.exp(-x))**2 )

def sigmoid_double_deriv( x ):
    return ( np.exp(-2.0*x) - np.exp(-x) )/(1.0+np.exp(-x))**3

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

    def maximum_cutoff_radius( self, atoms ):
        """
        Returns the maximum cutoff radius to avoid double interaction
        """
        cell = atoms.get_cell()
        v1 = cell[0,:]
        v2 = cell[1,:]
        v3 = cell[2,:]
        v1v2 = v1+v2
        v1v3 = v1+v3
        v2v3=v2+v2
        v1v2v3 = v1+v2+v3
        return 0.0

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

    def grad_force_single_atom_wrt_weights( self, inputs, grad_inp ):
        """
        Compute the gradient of the forces with respect to the weights
        """
        x = self.W.dot(inputs)
        current = 0
        grad = np.zeros( (3,self.total_number_of_weights()) )
        for i in range(self.W.shape[0]):
            for j in range(3):
                wdot_g_inp = self.W.dot(grad_inp[j,:])
                #term1 = sigmoid_double_deriv(x)*inputs*self.W.dot(grad_inp[j,:])
                term1 = sigmoid_double_deriv(x[i])*inputs*wdot_g_inp[i]
                term2 = sigmoid_deriv(x[i])*grad_inp[j,:]
                grad[j,current:current+self.W.shape[1]] = self.output_weights[i]*( term1 + term2 )
            current += self.W.shape[1]

        for j in range(3):
            grad[j,current:] = sigmoid_deriv(x)*self.W.dot(grad_inp[j,:])
        return grad

    def grad_total_energy_wrt_weights( self, atoms, nlist ):
        grad = None
        for i in range(len(atoms)):
            inputs = self.get_inputs( atoms, i, nlist )
            if ( grad is None ):
                grad = self.grad_energy_single_atom_wrt_weights(inputs)
            else:
                grad += self.grad_energy_single_atom_wrt_weights(inputs)
        return grad

    def grad_forces_wrt_weights( self, atoms, nlist ):
        """
        Computes the gradient of the forces with respect to the total weight
        """
        grad = []
        for i in range(len(atoms)):
            inputs = self.get_inputs( atoms, i, nlist )
            grad_inp = self.grad_inputs( atoms, i, nlist )
            grad.append( self.grad_force_single_atom_wrt_weights( inputs, grad_inp ) )
        return grad

    def get_neighbor_list( self, atoms ):
        """
        Returns the neighbor list required to evaluate the potential
        """
        cutoffs = [self.Rcut/2.0 for _ in range(len(atoms))]
        nlist = NeighborList(cutoffs,bothways=True,self_interaction=False,skin=0.0)
        nlist.update(atoms)
        return nlist

    def get_inputs( self, atoms, indx, nlist ):
        n_input_nodes = len(self.pairs)*self.n_sym_funcs_per_pair
        inputs = np.zeros(n_input_nodes)
        indices, offsets = nlist.get_neighbors(indx)
        mic_dist = self.offsets_to_mic_distance(atoms,indx,indices,offsets)
        lengths = np.sqrt( np.sum(mic_dist**2,axis=1) )
        for k,i in enumerate(indices):
            dist = lengths[k]
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

    def offsets_to_mic_distance( self, atoms, ref_atom, indices, offsets ):
        """
        Convert the offsets to minimum cell
        """
        mic_distance = np.zeros(offsets.shape)
        cell = atoms.get_cell()
        icell = np.linalg.pinv(cell)
        pos = atoms.get_positions()
        periodic_shift = offsets.dot(cell)
        r1 = pos[ref_atom,:]
        for i in range(offsets.shape[0]):
            indx = indices[i]
            r2 = pos[indx,:]+periodic_shift[i,:]
            mic_distance[i,:] = r2-r1
        return mic_distance

    def grad_inputs( self, atoms, indx, nlist ):
        n_input_nodes = len(self.pairs)*self.n_sym_funcs_per_pair
        grad_inp = np.zeros((3,n_input_nodes))
        indices, offsets = nlist.get_neighbors(indx)
        mic_distance = self.offsets_to_mic_distance( atoms, indx, indices, offsets )
        diff1 = np.sqrt(np.sum(mic_distance[0,:]**2))
        for k,i in enumerate(indices):
            dist = np.sqrt( np.sum(mic_distance[k,:]**2) )
            pair = self.create_pair_key( (atoms[indx].symbol,atoms[i].symbol) )
            sym_funcs = self.sym_funcs[pair]
            for sym_func in sym_funcs:
                grad_inp[:,sym_func.uid] += (self.cutoff_func.deriv(dist)*sym_func(dist) + sym_func.deriv(dist)*self.cutoff_func(dist))*mic_distance[k,:]/dist
        return grad_inp

    def get_force( self, atoms, indx, nlist ):
        """
        Compute the force on an atom (by central differences)
        """
        grad_inp = self.grad_inputs(atoms,indx,nlist)
        inputs = self.get_inputs(atoms,indx,nlist)
        x = self.W.dot(inputs)
        force = np.zeros(3)
        for i in range(3):
            neuron_output = sigmoid_deriv(x)*self.W.dot(grad_inp[i,:])
            force[i] = self.output_weights.dot(neuron_output)
        return force

    def get_forces( self, atoms, nlist=None ):
        """
        Compute the forces on all atoms in the structure
        """
        forces = np.zeros((len(atoms),3))
        if ( nlist is None ):
            nlist = self.get_neighbor_list(atoms)
        for i in range(len(atoms)):
            forces[i,:] = self.get_force( atoms, i, nlist )
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
        self.comm = None
        self.rank = 0

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
        # Distribute the weights from the master node
        if ( not self.comm is None ):
            weights = self.comm.bcast(weights,root=0)
        self.network.set_weights(weights)
        grad_E = np.zeros(len(weights))
        grad_F = np.zeros(len(weights))

        indx = range(len(self.structures))
        if ( not self.comm is None ):
            self.rank = self.comm.Get_rank()
            size = self.comm.Get_size()
            n_per_proc = float( len(self.structures) )/size
            start = int(n_per_proc*self.rank)
            end = int(n_per_proc*(self.rank+1))

            if ( self.rank == (size-1) ):
                indx = indx[start:]
            else:
                indx = indx[start:end]
        for i in indx:
            atoms = self.structures[i]
            ref_forces = atoms.get_forces()
            ref_energy = atoms.get_potential_energy()
            default_out_array = np.ones_like( ref_forces[:,0])

            if ( self.fit_energy ):
                tot_energy_nn = self.network.get_total_energy( atoms, nlist=self.nlists[i] )
                diff = ( (tot_energy_nn-ref_energy)/ref_energy )**2
                grad_E += self.grad_cost_func_energy_part_single_structure( tot_energy_nn, ref_energy, i )
                dE += diff
            if ( self.fit_forces ):
                N = ref_forces.shape[0]
                forces_nn = self.network.get_forces( atoms, nlist=self.nlists[i] )
                #dFx += np.sum( ( np.divide(forces_nn[:,0], ref_forces[:,0], where=ref_forces[:,0]!=0.0, out=default_out_array) - 1.0 )**2 )/N
                #dFy += np.sum( ( np.divide(forces_nn[:,1], ref_forces[:,1], where=ref_forces[:,1]!=0.0, out=default_out_array) - 1.0 )**2 )/N
                #dFz += np.sum( ( np.divide(forces_nn[:,2], ref_forces[:,2], where=ref_forces[:,2]!=0.0, out=default_out_array) - 1.0 )**2 )/N
                dFx += np.sum( (forces_nn[:,0] - ref_forces[:,0])**2 )
                dFy += np.sum( (forces_nn[:,1] - ref_forces[:,1])**2 )
                dFz += np.sum( (forces_nn[:,2] - ref_forces[:,2])**2 )
                grad_F += self.grad_cost_func_force_part_single_structure( forces_nn, ref_forces, i )

        # Sum contribution from each processor
        if ( not self.comm is None ):
            tot_dE = np.zeros(1)
            tot_grad_E = np.zeros(len(grad_E))
            tot_grad_F = np.zeros(len(grad_F))
            dFx_tot = np.zeros(1)
            dFy_tot = np.zeros(1)
            dFz_tot = np.zeros(1)
            self.comm.Allreduce(np.array(dE),tot_dE,op=MPI.SUM )
            self.comm.Allreduce(grad_E,tot_grad_E,op=MPI.SUM)
            self.comm.Allreduce(np.array(dFx),dFx_tot,op=MPI.SUM)
            self.comm.Allreduce(np.array(dFy),dFy_tot,op=MPI.SUM)
            self.comm.Allreduce(np.array(dFz),dFz_tot,op=MPI.SUM)
            self.comm.Allreduce(grad_F,tot_grad_F,op=MPI.SUM)
            dE = tot_dE[0]
            grad_E = tot_grad_E
            dFx = dFx_tot[0]
            dFy = dFy_tot[0]
            dFz = dFz_tot[0]
            grad_F = tot_grad_F

        avg_energy_diff = np.sqrt(dE)/len(self.structures)
        avg_Fx_diff = np.sqrt(dFx)/len(self.structures)
        avg_Fy_diff = np.sqrt(dFy)/len(self.structures)
        avg_Fz_diff = np.sqrt(dFz)/len(self.structures)
        cost = dE + (1.0/3.0)*( dFx + dFy + dFz )
        grad = grad_E + 2.0*self.lamb*weights/len(weights)**2

        if ( self.fit_forces ):
            grad += (1.0/3.0)*grad_F

        if ( self.rank == 0 ):
            print ("Energy difference: {}".format(avg_energy_diff))
            print ("Average force difference: {}".format([avg_Fx_diff,avg_Fy_diff,avg_Fz_diff]) )
            print ("Current cost function: {}".format(cost/len(self.structures)) )
            grad_norm = np.sqrt( np.sum( (grad/len(self.structures))**2 ) )
            print ("Norm of gradient: {}".format(grad_norm))
        return cost/len(self.structures) + self.lamb*self.penalization(), grad/len(self.structures)

    def grad_cost_func_energy_part_single_structure( self, tot_energy_nn, E_ref, struct_indx ):
        grad = self.network.grad_total_energy_wrt_weights( self.structures[struct_indx], self.nlists[struct_indx] )/E_ref
        return 2.0*(tot_energy_nn-E_ref)*grad/E_ref

    def grad_cost_func_force_part_single_structure( self, forces_nn, force_ref, struct_indx ):
        grad = self.network.grad_forces_wrt_weights( self.structures[struct_indx], self.nlists[struct_indx] )
        res = np.zeros(grad[0].shape[1])
        default_out = np.ones_like(forces_nn)
        #rel_dev = np.divide( forces_nn, force_ref, where=force_ref!=0.0, out=default_out )-1.0
        #rel_dev = np.divide( rel_dev, force_ref, where=force_ref!=0.0, out=np.zeros_like(rel_dev) )
        diff = forces_nn-force_ref
        # Loop over atoms
        for i in range(len(grad)):
            res += 2.0*diff[i,:].dot(grad[i])
        return res

    def penalization( self ):
        """
        Returns the penalization term
        """
        weights = np.array( self.network.get_weights() )
        return np.sum(weights**2)/len(weights)**2

    def train( self, method="BFGS", outfile="nnweights.csv", print_msg=True, comm=None, tol=1E-3 ):
        """
        Training the network
        """
        self.comm = comm
        if ( not self.comm is None ):
            self.rank = self.comm.Get_rank()
            if ( self.rank == 0 ):
                print ("Number of processors {}".format(self.comm.Get_size()))
        options = {
            "disp":print_msg,
            "eps":0.5
        }
        self.find_reasonable_initial_weights()
        x0 = self.network.get_weights()
        res = minimize( self.cost_function, x0, method=method, jac=True, options=options, tol=tol )
        if ( self.rank == 0 ):
            ts = time.strftime("%Y%m%d_%H%M%S")
            splitted = outfile.split(".")
            fname = splitted[0] + "_%s."%(ts) + splitted[1]
            header = "Pairs: {}\n".format(self.network.pairs)
            header += "Number of symmetry functions per pair: {}\n".format(self.network.n_sym_funcs_per_pair)
            header += "Number of hidden layers: {}".format( len(self.network.output_weights) )
            np.savetxt(fname, res["x"], delimiter=",", header=header )
            print ( "Neural network weights written to %s"%(fname) )

    def plot_energy( self, test_structures=None ):
        """
        Generates a plot comparing the DFT energies and the energies from the
        pair potential
        """
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        energies_dft = [struct.get_potential_energy()/len(struct) for struct in self.structures]
        energies_nn = [self.network.get_total_energy(struct, nlist=nlist)/len(struct) for struct,nlist in zip(self.structures,self.nlists)]
        ax.plot( energies_dft, energies_dft )
        ax.plot( energies_nn, energies_dft, "o", mfc="none", label="Training data" )

        if ( not test_structures is None ):
            edft_test = [struct.get_potential_energy()/len(struct) for struct in test_structures]
            e_nn_test = [self.network.get_total_energy(struct)/len(struct) for struct in test_structures]
            ax.plot( edft_test, e_nn_test, "o", mfc="none", label="Test data" )
        ax.set_xlabel( "Energy NN potential (eV/atom)" )
        ax.set_ylabel( "Energy DFT (eV/atom)" )
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.legend( loc="best", frameon=False )
        return fig

    def plot_forces( self, test_structures=None ):
        """
        Creates a plot to see the convergence of the forces
        """
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        forces_dft = [[],[],[]]
        forces_nn = [[],[],[]]
        for struct,nlist in zip(self.structures,self.nlists):
            f_dft = struct.get_forces()
            f_nn = self.network.get_forces(struct,nlist=nlist)
            for i in range(3):
                forces_dft[i] += list(f_dft[:,i])
                forces_nn[i] += list(f_nn[:,i])

        labels=["x","y","z"]
        ax.plot( forces_dft[0], forces_dft[0] )
        for i in range(3):
            ax.plot( forces_nn[i], forces_dft[i], "o", mfc="none", label=labels[i] )
        ax.set_xlabel( "Forces NN (eV/A)" )
        ax.set_ylabel( "Forces DFT (eV/A)" )
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.legend( loc="best", frameon=False )
        return fig
