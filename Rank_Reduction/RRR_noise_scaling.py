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



def get_min_test_loss_for_top_k(val_losses, test_losses, k=3):
    if len(val_losses) != len(test_losses) or len(val_losses) < k:
        raise ValueError("Invalid input lengths or k too large")
    paired = sorted(zip(val_losses, test_losses), key=lambda x: x[0])
    return min(test_loss for (val_loss, test_loss) in paired[:k])

def RRR(train_dataset, val_dataset, test_dataset, input_dim, output_dim, IN = True, bias = False):

    W_, _, _ = svd_mix_ch_linear(train_dataset, test_dataset, instance_norm=IN, bias=bias)

    test_mse_listw = []
    val_mse_listw = []
    rsw = []
    use_bias=False
    
    y_train_hat = obtain_y_hat_mix_ch(        
        train_dataset, 
        W_,
        instance_norm = IN, 
        bias = bias
    )
    _, _, Vt_y = np.linalg.svd(y_train_hat, full_matrices=False)
    del y_train_hat
    
    for r in range(6, output_dim+6, 6):
        if r > output_dim:
            break
        
        # uw, sw, vtw = np.linalg.svd(W_, full_matrices=False)
        # sw[r:] = 0
        # W_redu = uw @ np.diag(sw) @ vtw
        W_redu = W_ @ Vt_y[:r, :].T @ Vt_y[:r, :]
    
        res_, preds_ = infer_ols_mixed_ch(
            test_dataset=test_dataset,
            W_ = W_redu,
            instance_norm=IN,
            bias=bias
        )
        res_val, preds_val = infer_ols_mixed_ch(
            test_dataset=val_dataset,
            W_ = W_redu,
            instance_norm=IN,
            bias=bias
        )
        
        test_mse,_ = res_
        val_mse,_= res_val
        test_mse_listw.append(test_mse)
        val_mse_listw.append(val_mse)
        rsw.append(r)
    
    # title = f"{path_} L{input_dim} H{o_dim} RRR"
    # save_path = f"./RRR/{path_}_L{input_dim}_H{o_dim}_RRR.png"
    # plot_dual_axis_curve(rsw, val_mse_listw, test_mse_listw, title=title, save_path=save_path)
    print(f"optimal val idx: {np.argmin(val_mse_listw)}, optimal test idx: {np.argmin(test_mse_listw)}")
    print(f"test loss on min val: {test_mse_listw[np.argmin(val_mse_listw)]:.3f}")
    print(f"test loss on min 3 val: {get_min_test_loss_for_top_k(val_mse_listw, test_mse_listw, 3):.3f}")
    print(f"test loss on min 5 val: {get_min_test_loss_for_top_k(val_mse_listw, test_mse_listw, 5):.3f}")
    print(f"min test loss {np.min(test_mse_listw):.3f}")
    return get_min_test_loss_for_top_k(val_mse_listw, test_mse_listw, 3)

input_dim = 720
output_dim = 720
all_results = []
for repeat in range(5):
    results = []
    for sigma in [0, 0.25, 0.5, 0.75, 1]:
        def simulation_func_n(x):
            return np.sin(2*x) + np.cos(5*x) + 0.5*x + np.random.normal(loc=0.0, scale=1.0, size=x.shape)*sigma
        data_core_n = Dataset_Function_MC_Core(functions = [simulation_func_n], random_generator=None, size = [input_dim, 0, output_dim], scale=False, x_end=400)
        train_dataset_n = Dataset_Function_MC(data_core_n, flag="train")
        val_dataset_n = Dataset_Function_MC(data_core_n, flag="val")
        test_dataset_n = Dataset_Function_MC(data_core_n, flag="test")
        
        
        
        
        score_ = RRR(train_dataset_n, val_dataset_n, test_dataset_n, input_dim = input_dim, output_dim = output_dim, IN = True, bias = False)
        results.append(score_)
    all_results.append(results)

all_results = np.array(all_results)
mean_performance = all_results.mean(0)
std_performance = all_results.std(0)

for i, sigma in enumerate([0, 0.25, 0.5, 0.75, 1]):
    print(f"data noise: {sigma}, mean mse: {mean_performance[i]:.3f} ± {std_performance[i]:.3f}")
