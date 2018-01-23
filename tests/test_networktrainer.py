import unittest
from nnpotential import neural_network as nn
from ase.build import bulk
from ase.calculators.emt import EMT

class TestNetworkTrainer( unittest.TestCase ):
    def test_one_iteration( self ):
        no_exception = True
        try:
            network = nn.NNPotential( Rcut=4.1, pairs=["Al-Al"] )
            atoms = bulk("Al","fcc",a=4.05)
            atoms = atoms*(4,4,4)
            calc = EMT()
            atoms.set_calculator(calc)
            trainer = nn.NetworkTrainer( [atoms], network )
            trainer.train( tol=1E10 ) # Set a large tolererance such that it converges after one iteration
        except Exception as exc:
            print (str(exc))
            no_exception = False

        self.assertTrue( no_exception )

if __name__ == "__main__":
    unittest.main()
