import sys
from nnpotential import neural_network as nn
from nnpotential import structure_provider as sp
import numpy as np
import matplotlib as mpl
mpl.rcParams["svg.fonttype"] = "none"
mpl.rcParams["font.size"] = 18
mpl.rcParams["axes.unicode_minus"] = False
from matplotlib import pyplot as plt
from mpi4py import MPI

comm = MPI.COMM_WORLD

db_name = "data/almg_structures.db"
def main( argv ):
    opt = argv[0]
    sym_width = float(argv[1])
    provider = sp.StructureProvider( db_name )
    potential = nn.NNPotential( pairs=["Al-Al","Al-Mg","Mg-Mg"], n_sym_funcs_per_pair=20, sym_func_width=sym_width, Rcut=4.1, Rmin=1.0,n_hidden=30 )
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

    if ( opt == "train" ):
        trainer = nn.NetworkTrainer( filtered_structs, potential, lamb=0.0, fit_forces=True, fit_energy=False )
        trainer.train( method="BFGS", outfile="data/nn_almg_weights_with_force.csv", comm=comm, tol=1E-4 )
    elif ( opt == "eval" ):
        evaluate( potential, structures, "data/nn_almg_weights_with_force_20180124_171211.csv", "data/control_indices.csv" )

def evaluate( network, all_structs, weight_file, control_file ):
    control_indices = np.loadtxt( control_file, delimiter=",").astype(np.int32)
    weights = np.loadtxt( weight_file, delimiter=",")
    network.set_weights(weights)
    network.plot_weights()
    filtered_structs = []
    test_structures = []
    for i in range(len(all_structs)):
        if ( i in control_indices ):
            test_structures.append(all_structs[i])
        else:
            filtered_structs.append(all_structs[i])
    trainer = nn.NetworkTrainer( filtered_structs, network )
    trainer.plot_energy( test_structures=test_structures )
    trainer.plot_forces()
    plt.show()


if __name__ == "__main__":
    main( sys.argv[1:])
