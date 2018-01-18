from nnpotential import neural_network as nn
from nnpotential import structure_provider as sp
import numpy as np

db_name = "data/almg_structures.db"
def main():
    provider = sp.StructureProvider( db_name )
    potential = nn.NNPotential( pairs=["Al-Al","Al-Mg","Mg-Mg"], n_sym_funcs_per_pair=20, sym_func_width=1.5, Rcut=6.0, Rmin=1.0,n_hidden=30 )
    structures = provider.get()

    # Select 50 random structures to be used as tests
    rand_structs = np.random.randint(low=0,high=len(structures),size=50)
    np.savetxt("data/control_indices.csv", rand_structs, delimiter="," )
    filtered_structs = []
    for indx in range(len(structures)):
        if ( indx in rand_structs ):
            continue
        else:
            filtered_structs.append(structures[indx])
    trainer = nn.NetworkTrainer( filtered_structs, potential, lamb=0.0, force_step_size=0.04, fit_forces=False )
    trainer.train( method="BFGS", outfile="data/nn_almg_weights.csv" )

if __name__ == "__main__":
    main()
