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

if __name__ == "__main__":
    main()
