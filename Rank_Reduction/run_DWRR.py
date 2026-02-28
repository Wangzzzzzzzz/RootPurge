import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from data_provider import data_factory
from copy import deepcopy
import pickle as pkl

from solver.ols import infer_ols_mixed_ch, infer_ols_indp_ch, svd_indp_ch_linear, svd_mix_ch_linear

def plot_dual_axis_curve(x, y1, y2, label1="Val MSE", label2="Test MSE", color1="#1f77b4", color2="#ff7f0e", title="Test vs Val", save_path="result.png"):
    plt.figure(figsize=(10, 8))
    fig, ax1 = plt.subplots()
    
    # First curve
    ax1.plot(x, y1, color=color1, label=label1)
    ax1.set_xlabel("Rank of $W$")
    ax1.set_ylabel(label1, color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)
    
    # Second axis for the second curve
    ax2 = ax1.twinx()
    ax2.plot(x, y2, color=color2, label=label2)
    ax2.set_ylabel(label2, color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)
    
    plt.title(title)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def get_min_test_loss_for_top_k(val_losses, test_losses, k=3):
    if len(val_losses) != len(test_losses) or len(val_losses) < k:
        raise ValueError("Invalid input lengths or k too large")
    paired = sorted(zip(val_losses, test_losses), key=lambda x: x[0])
    return min(test_loss for (val_loss, test_loss) in paired[:k])

file_names = ["ETT-small","ETT-small","ETT-small","ETT-small","electricity","weather","traffic","exchange_rate"]
pathes = ["ETTh1","ETTh2","ETTm1","ETTm2","electricity","weather","traffic","exchange_rate"]
input_dim = 720


for f_name, path_ in zip(file_names,pathes):
    for o_dim in [96, 192, 336, 720]:
        print(f"data: {path_}, input: {input_dim}, output: {o_dim}")
        if f_name != "ETT-small":
            train_dataset, train_data_loader = data_factory.data_provider(
                data = "custom",
                batch_size=128,
                root_path=f"datasets/time_series/{f_name}",
                data_path=f"{path_}.csv",
                size = [input_dim, 0, o_dim],
                flag = "train"
            )
            
            val_dataset, val_data_loader = data_factory.data_provider(
                data = "custom",
                batch_size=32,
                root_path=f"datasets/time_series/{f_name}",
                data_path=f"{path_}.csv",
                size = [input_dim, 0, o_dim],
                flag = "val"
            )
            
            test_dataset, test_data_loader = data_factory.data_provider(
                data = "custom",
                batch_size=32,
                root_path=f"datasets/time_series/{f_name}",
                data_path=f"{path_}.csv",
                size = [input_dim, 0, o_dim],
                flag = "test"
            )

        else:
            train_dataset, train_data_loader = data_factory.data_provider(
                data = path_,
                batch_size=256,
                root_path=f"datasets/time_series/{f_name}",
                data_path=f"{path_}.csv",
                size = [input_dim, 0, o_dim],
                flag = "train"
            )

            val_dataset, val_data_loader = data_factory.data_provider(
                data = path_,
                batch_size=32,
                root_path=f"datasets/time_series/{f_name}",
                data_path=f"{path_}.csv",
                size = [input_dim, 0, o_dim],
                flag = "val"
            )

            test_dataset, test_data_loader = data_factory.data_provider(
                data = path_,
                batch_size=32,
                root_path=f"datasets/time_series/{f_name}",
                data_path=f"{path_}.csv",
                size = [input_dim, 0, o_dim],
                flag = "test"
            )
        
        if path_ == "weather":
            W_, _, _ = svd_indp_ch_linear(train_dataset, test_dataset, instance_norm=True, bias=False)
        
            W_ = W_.transpose(2,0,1) # toc -> cto
            uw, sw_, vtw = np.linalg.svd(W_, full_matrices=False)

            test_mse_listw = []
            val_mse_listw = []
            rsw = []
            use_bias=False
            
            for r in range(6, o_dim+6, 6):
                if r > o_dim:
                    break
                    
                sw = deepcopy(sw_)
                sw[:,r:] = 0
                W_redu = []
                for i in range(uw.shape[0]):
                    W_redu.append(uw[i] @ np.diag(sw[i]) @ vtw[i])
                W_redu = np.array(W_redu)
                W_redu = np.ascontiguousarray(W_redu.transpose(1,2,0))
                # W_redu = W_ @ vtw[:r, :].T @ vtw[:r, :]
            
                res_, preds_ = infer_ols_indp_ch(
                    test_dataset=test_dataset,
                    W_ = W_redu,
                    instance_norm=True,
                    bias=use_bias
                )
                res_val, preds_val = infer_ols_indp_ch(
                    test_dataset=val_dataset,
                    W_ = W_redu,
                    instance_norm=True,
                    bias=use_bias
                )
                
                test_mse,_ = res_
                val_mse,_= res_val
                test_mse_listw.append(test_mse)
                val_mse_listw.append(val_mse)
                rsw.append(r)
        else:
            W_, _, _ = svd_mix_ch_linear(train_dataset, test_dataset, instance_norm=True, bias=False)
    
            uw, sw_, vtw = np.linalg.svd(W_, full_matrices=False)
            test_mse_listw = []
            val_mse_listw = []
            rsw = []
            use_bias=False
            
            for r in range(6, o_dim+6, 6):
                if r > o_dim:
                    break
                
                sw = deepcopy(sw_)
                sw[r:] = 0
                W_redu = uw @ np.diag(sw) @ vtw
                # W_redu = W_ @ vtw[:r, :].T @ vtw[:r, :]
            
                res_, preds_ = infer_ols_mixed_ch(
                    test_dataset=test_dataset,
                    W_ = W_redu,
                    instance_norm=True,
                    bias=use_bias
                )
                res_val, preds_val = infer_ols_mixed_ch(
                    test_dataset=val_dataset,
                    W_ = W_redu,
                    instance_norm=True,
                    bias=use_bias
                )
                
                test_mse,_ = res_
                val_mse,_= res_val
                test_mse_listw.append(test_mse)
                val_mse_listw.append(val_mse)
                rsw.append(r)

        dwrr_raw_results = {
            "test_mse_listw":test_mse_listw,
            "val_mse_listw":val_mse_listw,
            "rsw":rsw
        }

        with open(f"./DWRR_raw_results/{path_}_L{input_dim}_H{o_dim}_DWRR.pkl", "wb") as f:
            pkl.dump(dwrr_raw_results, f)

        title = f"{path_} L{input_dim} H{o_dim} DWRR"
        save_path = f"./DWRR/{path_}_L{input_dim}_H{o_dim}_DWRR.png"
        plot_dual_axis_curve(rsw, val_mse_listw, test_mse_listw, title=title, save_path=save_path)
        print(f"optimal val idx: {np.argmin(val_mse_listw)}, optimal test idx: {np.argmin(test_mse_listw)}")
        print(f"test loss on min val: {test_mse_listw[np.argmin(val_mse_listw)]:.3f}")
        print(f"test loss on min 3 val: {get_min_test_loss_for_top_k(val_mse_listw, test_mse_listw, 3):.3f}")
        print(f"test loss on min 5 val: {get_min_test_loss_for_top_k(val_mse_listw, test_mse_listw, 5):.3f}")
        print(f"min test loss {np.min(test_mse_listw):.3f}")