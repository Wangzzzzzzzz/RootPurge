import numpy as np

import random
from solver.ols import infer_ols_mixed_ch, svd_mix_ch_linear, obtain_y_hat_mix_ch
from data_provider.data_func import Dataset_Function_MC_Core, Dataset_Function_MC

# fix_seed = 0
# random.seed(fix_seed)
# np.random.seed(fix_seed)

def simulation_func_n(x):
    return np.sin(2*x) + np.cos(5*x) + 0.5*x + np.random.normal(loc=0.0, scale=1.0, size=x.shape)*0.5


def simulation_func_nf(x):
    return np.sin(2*x) + np.cos(5*x) + 0.5*x


input_dim = 720
output_dim = 720

all_results = []
for repeat in range(5):
    results = []
    for ending in [100, 200, 400, 800, 1600]:
        data_core_n = Dataset_Function_MC_Core(functions = [simulation_func_n], random_generator=None, size = [input_dim, 0, output_dim], scale=False, x_end=ending)
        train_dataset_n = Dataset_Function_MC(data_core_n, flag="train")
        val_dataset_n = Dataset_Function_MC(data_core_n, flag="val")
        test_dataset_n = Dataset_Function_MC(data_core_n, flag="test")
        
        
        # data_core_nf = Dataset_Function_MC_Core(functions = [simulation_func_nf], random_generator=None, size = [input_dim, 0, output_dim], scale=False, x_end=ending)
        # train_dataset_nf = Dataset_Function_MC(data_core_nf, flag="train")
        # val_dataset_nf = Dataset_Function_MC(data_core_nf, flag="val")
        # test_dataset_nf = Dataset_Function_MC(data_core_nf, flag="test")
        
        
        W_n, pred_n, scores_n = svd_mix_ch_linear(train_dataset_n, 
                                               test_dataset_n, 
                                               instance_norm = True, 
                                               bias = False
                                              )
        results.append(scores_n[0])
    all_results.append(results)

all_results = np.array(all_results)
#print(all_results)
mean_performance = all_results.mean(0)
std_performance = all_results.std(0)

for i, L in enumerate([100, 200, 400, 800, 1600]):
    print(f"data length: {L*100}, mean mse: {mean_performance[i]:.3f} ± {std_performance[i]:.3f}")