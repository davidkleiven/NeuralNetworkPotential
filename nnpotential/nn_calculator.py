from ase.calculators.calculator import Calculator, all_changes
import numpy as np

class NN( Calculator ):
    implemented_properties = ["energy","forces","energies"]
    def __init__( self, network=None, **kwargs ):
        Calculator.__init__( self, **kwargs )
        if ( network is None ):
            raise ValueError( "A neural network potential object has to be passed" )
        self.network = network
        self.nlsit = None

        # Check if an atoms object is passed
        for key,value in kwargs.iteritems():
            if ( key=="atoms" ):
                self.atoms = value
                self.nlist = self.network.get_neighbor_list(self.atoms)

    def calculate(self,atoms=None, properties=["energy"], system_changes=all_changes):
        Calculator.calculate(self,atoms,properties,system_changes)
        if ( ("positions" in all_changes) or ("cell" in all_changes) ):
            # TODO: Check if the positions have moved less than the skin-depth
            # and only update neighbor list if atoms have moved a significant amount
            self.nlist = self.network.get_neighbor_list(self.atoms)

        if ( ("energy" in properties) or ("energies" in properties) ):
            self.results["energies"] = [self.network.get_potential_energy( self.atoms, indx, self.nlist ) for indx in range(len(self.atoms))]
            self.results["energy"] = np.sum(self.results["energies"])

        if ( "forces" in properties ):
            self.results["forces"] = self.network.get_forces( self.atoms, nlist=self.nlist )

    def get_potential_energies( self, atoms=None ):
        return self.get_property( "energies", atoms=atoms )
