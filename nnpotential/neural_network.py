import numpy as np
from scipy.spatial import cKDTree as KDTree
from scipy.optimize import minimize

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
        self.weights = np.ones(n_inputs)

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

    def total_number_of_weights( self ):
        return len(self.pairs)*self.n_sym_funcs_per_pair*len(self.hidden_neurons) + len(self.hidden_neurons)

    def get_weights( self ):
        weights = []
        for neuron in self.hidden_neurons:
            weights += list(neuron.weights)
        weights += list(self.output_neuron.weights)
        return weights

    def set_weights( self, weights ):
        if ( len(weights) != self.total_number_of_weights() ):
            raise ValueError( "The number of weights does not match the required number" )

        current = 0
        for neuron in self.hidden_neurons:
            neuron.weights[:] = weights[current:current+len(neuron.weights)]
            current += len(neuron.weights)
        self.output_neuron.weights[:] = weights[current:]

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

    def get_all_forces( self, atoms, step_size=0.04 ):
        """
        Compute the forces on all atoms in the structure
        """
        forces = np.zeros((len(atoms),3))
        for i in range(len(atoms)):
            forces[i,:] = self.get_force_vector( atoms,i, step_size )
        return forces


    def get_total_energy( self, atoms ):
        """
        Computes the total energy of the system
        """
        tot_energy = 0.0
        for i in range(len(atoms)):
            tot_energy += self.get_potential_energy( atoms, i )
        return tot_energy

class NetworkTrainer( object ):
    """
    Class for training the network
    """
    def __init__( self, structures, network, lamb=1.0, force_step_size=0.04 ):
        self.structures = structures
        self.network = network
        self.lamb = lamb
        self.force_step_size = force_step_size

    def cost_function( self, weights ):
        dE = 0.0
        dFx = 0.0
        dFy = 0.0
        dFz = 0.0
        self.network.set_weights(weights)
        for atoms in self.structures:
            ref_forces = atoms.get_forces()
            ref_energy = atoms.get_potential_energy()
            mean_force = np.mean( np.abs(ref_forces),axis=0 )

            tot_energy_nn = self.network.get_total_energy( atoms )
            forces_nn = self.network.get_forces( atoms, step_size=force_step_size )
            dE += ( (tot_energy_nn-ref_energy)/ref_energy )**2
            dFx += np.mean( (forces_nn[:,0]-ref_forces[:,0])**2 )/mean_force[0]**2
            dFy += np.mean( (forces_nn[:,1]-ref_forces[:,1])**2 )/mean_force[1]**2
            dFz += np.mean( (forces_nn[:,2]-ref_forces[:,2])**2 )/mean_force[2]**2

        cost = dE + (1.0/(3.0)*( dFx + dFy + dFz )
        return cost + self.lamb*self.penalization()

    def penalization( self ):
        """
        Returns the penalization term
        """
        sum_sq_weights = 0.0
        for neuron in self.network.hidden_neurons:
            sum_sq_weights += np.sum(neuron.weights**2)
        sum_sq_weights += np.sum(self.output_neuron.weights**2)
        n_weights = self.network.total_number_of_weights()
        return sum_sq_weights/n_weights**2

    def train( self, method="BFGS", outfile="nnweights.csv" ):
        """
        Training the network
        """
        x0 = self.network.get_weights()
        res = minimize( self.cost_function, x0, method=method )
        np.savetxt(outfile, res["x"], delimiter=",")
        print ( "Neural network weights written to %s"%(outfile) )
