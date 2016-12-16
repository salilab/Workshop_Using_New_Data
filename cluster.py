"""Filter and cluster replicas."""
import glob
import IMP
import IMP.pmi
import IMP.pmi.macros

n_models = 1000
n_clusters = 5
prot_name = '1G6J'
merged_dirs = glob.glob("output*/")

m = IMP.Model()
mc = IMP.pmi.macros.AnalysisReplicaExchange0(m,
                                             stat_file_name_suffix="stat",
                                             merge_directories=merged_dirs,
                                             global_output_directory='./')

feature_list = ['Total_Score', 'NOE_Sigma', 'NOE_Gamma', 'Dihedral_Kappa',
                'CHARMM_BONDS', 'CHARMM_NONBONDED', 'NOELikelihood_Score',
                'NOEPrior_Score', 'DihedralLikelihood_Score',
                'DihedralPrior_Score', 'rmf_frame_index']

mc.clustering("Total_Score",
              "rmf_file",
              "rmf_frame_index",
              rmsd_calculation_components={prot_name: prot_name},
              alignment_components={prot_name: prot_name},
              number_of_clusters=n_clusters,
              feature_keys=feature_list,
              number_of_best_scoring_models=n_models,
              distance_matrix_file="distance.rawmatrix.pkl",
              skip_clustering=False,
              get_every=1)
