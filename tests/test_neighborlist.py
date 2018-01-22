import unittest
from ase.build import bulk
from nnpotential import neural_network as nn

class TestNeighborList( unittest.TestCase ):
    def test_neighborlist(self):
        Rcut = 5.73 # Should cover third nearest neighbors on Al FCC
        pot = nn.NNPotential( Rcut=Rcut )
        atoms = bulk("Al","fcc",a=4.05)
        nlist = pot.get_neighbor_list(atoms)
        indices, offsets = nlist.get_neighbors(0)

        # 5.73 should cover third nearest neighbors
        n_nearest_neighbors = 12
        n_second_nearest_neighbors = 6
        n_third = 24
        n_neighbors = n_nearest_neighbors+n_second_nearest_neighbors+n_third
        self.assertEqual( len(indices), n_neighbors )

        mic_distance = pot.offsets_to_mic_distance( atoms, 0, indices, offsets )
        lengths = np.sqrt(np.sum(mic_distance**2,axis=1))
        for i in range(len(lengths)):
            self.assertTrue( lenghts[i] < Rcut )

if __name__ == "__main__":
    unittest.main()
