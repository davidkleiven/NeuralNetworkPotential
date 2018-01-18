import sys
from nnpotential import structure_provider as stprov
from ase.io.trajectory import Trajectory
import glob

db_name = "data/almg_structures.db"
trajectory_folder = "/home/davidkl/Documents/TrajectoryFiles/Trajectory_AlMg"

def main():
    provider = stprov.StructureProvider( db_name )
    all_files = glob.glob(trajectory_folder+"/*.traj")
    for fname in all_files:
        traj = Trajectory(fname)
        provider.insert(traj)

if __name__ == "__main__":
    main()
