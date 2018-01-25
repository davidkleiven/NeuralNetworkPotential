import unittest
from nnpotential import neural_network as nn
from ase.build import bulk
import numpy as np
import copy

class TestGradients( unittest.TestCase ):
    def test_grad_wrt_weights(self):
        pot = nn.NNPotential( pairs=["Al-Al","Al-Mg","Mg-Mg"], Rcut=4.1, n_sym_funcs_per_pair=10 )
        atoms = bulk("Al")
        atoms = atoms*(4,4,4)
        nlist = pot.get_neighbor_list(atoms)
        init_weights = np.loadtxt( "tests/nn_almg_weights.csv" )
        pot.set_weights(init_weights)
        weights = pot.get_weights()
        grad_weights = pot.grad_total_energy_wrt_weights( atoms, nlist )
        delta = 0.000000005
        orig_energy = pot.get_total_energy( atoms, nlist=nlist )
        num_delta = []
        for i in range(len(weights)):
            weights_copy = copy.deepcopy(weights)
            weights_copy[i] += delta
            pot.set_weights(weights_copy)
            num_delta = pot.get_total_energy(atoms,nlist=nlist)-orig_energy
            num_deriv = num_delta/delta
            self.assertAlmostEqual( grad_weights[i], num_deriv, places=3 )

    def test_forces(self):
        pot = nn.NNPotential( pairs=["Al-Al","Al-Mg","Mg-Mg"], Rcut=4.1, n_sym_funcs_per_pair=10 )
        atoms = bulk("Al")
        atoms = atoms*(4,4,4)
        atoms.rattle()
        init_weights = np.loadtxt( "tests/nn_almg_weights.csv" )
        pot.set_weights(init_weights)
        n_atoms_to_test_forces = 40
        nlist = pot.get_neighbor_list(atoms)
        forces_nn = pot.get_forces(atoms, nlist=nlist ) # Neighbor list is built internally
        orig_pos = atoms.get_positions()
        delta = 0.001
        for i in range(n_atoms_to_test_forces):
            nlist = pot.get_neighbor_list(atoms)
            for j in range(3):
                pos_cpy = copy.deepcopy(orig_pos)
                pos_cpy[i,j] += delta/2.0
                atoms.set_positions(pos_cpy)
                nlist = pot.get_neighbor_list(atoms)
                Epluss = pot.get_potential_energy( atoms, i, nlist )
                pos_cpy[i,j] -= delta
                atoms.set_positions(pos_cpy)
                nlist = pot.get_neighbor_list(atoms)
                Eminus = pot.get_potential_energy( atoms, i, nlist )
                force = -(Epluss-Eminus)/delta # Force = negative gradient of the energy
                self.assertLess( np.abs( forces_nn[i,j]/force - 1.0), 0.001 )
            atoms.set_positions(orig_pos)

    def test_force_grad_wrt_weights(self):
        pot = nn.NNPotential( pairs=["Al-Al","Al-Mg","Mg-Mg"], Rcut=4.1, n_sym_funcs_per_pair=10 )
        atoms = bulk("Al")
        atoms = atoms*(4,4,4)
        atoms.rattle()
        init_weights = np.loadtxt( "tests/nn_almg_weights.csv" )
        pot.set_weights(init_weights)
        nlist = pot.get_neighbor_list(atoms)
        weights = pot.get_weights()
        delta = 0.0000005
        grad_f = pot.grad_forces_wrt_weights( atoms, nlist )
        forces = pot.get_forces( atoms, nlist=nlist )
        for i in range(len(weights)):
            weights_cpy = copy.deepcopy(weights)
            weights_cpy[i] += delta
            pot.set_weights(weights_cpy)
            forces_pluss = pot.get_forces( atoms, nlist )
            num_grad = (forces_pluss-forces)/delta

            # Loop over atoms
            for j in range(num_grad.shape[0]):
                exc_grad = grad_f[j]
                # Loop over components
                for k in range(num_grad.shape[1]):
                    #print (exc_grad[k,i], num_grad[j,k])
                    if ( exc_grad[k,i] == 0.0 or num_grad[j,k] == 0.0 ):
                        self.assertAlmostEqual( exc_grad[k,i], num_grad[j,k], places=7 )
                    else:
                        self.assertLess( np.abs(exc_grad[k,i]/num_grad[j,k]-1.0), 0.001 )

    def test_load_save(self):
        pot = nn.NNPotential( pairs=["Al-Al","Al-Mg","Mg-Mg"], Rcut=4.1, n_sym_funcs_per_pair=10, Rmin=0.5 )
        init_weights = np.loadtxt( "tests/nn_almg_weights.csv" )
        pot.set_weights(init_weights)
        fname = "test_save.json"
        pot.save( fname )
        loaded_pot =nn. NNPotential.load(fname)
        self.assertTrue( pot==loaded_pot )

if __name__ == "__main__":
    unittest.main()
