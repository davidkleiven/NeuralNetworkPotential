from ase.db import connect

class StructureProvider( object ):
    """
    Class that manages the structures to be used in the training
    """
    def __init__( self, db_name ):
        self.db = connect(db_name)
        self.db_energies = None

    def get( self, selection="" ):
        """
        Returns a list of all structures in the database matching the selection criteria
        """
        structures = []
        for row in self.db.select( selection=selection ):
            structures.append( self.db.get_atoms(id=row.id) )
        return structures

    def insert( self, new_structures, tol=8 ):
        """
        Insert new structures in to the database

        new_structures - list of new structures
        tol - if more than tol digits of the energy of two structures are equal
              they are considered duplicates
        """
        if ( self.db_energies is None ):
            self.read_db_energies()

        num_inserted = 0
        for struct in new_structures:
            energy = struc.get_potential_energy()
            if ( not self.exists_in_db(energy,tol) ):
                self.db.write( struct )
                self.db_energies.append(energy)
                num_inserted += 1

        print ("Inserted {} structures into the database".format(num_inserted) )

    def read_db_energies( self ):
        """
        Reads all the energies currently stored in the database
        """
        for row in self.db.select():
            db_energy = row.get("energy")
            if ( not db_energy is None ):
                self.db_energies.append()

    def exists_in_db( self, energy, n_digits ):
        """
        Check if an entry with this energy already exists in the database
        """
        if ( self.db_energies is None ):
            self.read_db_energies()
        factor = 10**n_digits
        for eng in energies:
            diff = abs(energy-eng)
            if ( int(diff*factor) == 0 ):
                return False
        return True
