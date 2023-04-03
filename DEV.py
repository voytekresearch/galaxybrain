
from galaxybrain.data_utils import load_results
from galaxybrain.plot_utils import plot_all_measures


sim_data = load_results('./data/experiments/ising', kind='sim')
plot_all_measures(sim_data['2.27'], sim_data['meta'])