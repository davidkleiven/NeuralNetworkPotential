import unittest
from nnpotential.nn_calculator import NN
from nnpotential.neural_network import NNPotential
from ase.build import bulk
import numpy as np

class TestCalc( unittest.TestCase ):
    def test_calc_no_throw(self):
        no_throw = True
        try:
            atoms = bulk("Al")
            atoms = atoms*(4,4,4)
            network = NNPotential( Rcut=4.1, pairs=["Al-Al","Al-Mg","Mg-Mg"],n_sym_funcs_per_pair=10 )
            weights = np.loadtxt( "tests/nn_almg_weights.csv" )
            network.set_weights(weights)
            calc = NN(network,atoms=atoms)
            atoms.set_calculator(calc)
            energy = atoms.get_potential_energy()
            energies = atoms.get_potential_energies()
            forces = atoms.get_forces()
        except Exception as exc:
            print (exc)
            no_throw = False
        self.assertTrue(no_throw)

if __name__ == "__main__":
    unittest.main()
