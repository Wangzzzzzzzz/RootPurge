from copy import deepcopy
from functools import partial
import os
import numpy as np

import torch
from torch import nn

from synthetic_data import Dataset_Function_MC_Core, Dataset_Function_MC
from genTS.external.tslib.utils.tools import EarlyStopping
from genTS.external.tslib.data_provider import data_factory as tslib_loader
from genTS.utils.argparser_default import get_parser
from genTS.utils.tools import str2bool
from torch.utils.data import Dataset, DataLoader
import random
import torch.nn.functional as F
from optimizer import WeightDecayOptimizer



class BaseLinearModel(nn.Module):
    def __init__(self, configs):
        ## ignore lin_prob since this is linear model
        super(BaseLinearModel, self).__init__()
        self.IN = configs.instance_norm
        self.indv = configs.individual
        if self.indv:
            self.l1 = nn.ModuleList(
                [nn.Linear(configs.seq_len, configs.pred_len, bias=False) for _ in range(configs.enc_in)]
            )
        else:
            self.l1 = nn.Linear(configs.seq_len, configs.pred_len, bias=False)

    def forward(self, x, *args):
        if self.IN:
            x_mean = x.mean(dim = 1, keepdim = True)
            x = x-x_mean
        x = x.permute(0,2,1) # x: [B, C, T]
        if self.indv:
            y = [l_c(x[:,[idx],:]) for idx, l_c in enumerate(self.l1)]
            y = torch.cat(y,dim=1).permute(0,2,1)
        else:
            y = self.l1(x).permute(0,2,1)
        
        if self.IN:
            y = y + x_mean
        return y


class SpecLinear(nn.Module):
    def __init__(self, configs):
        super(SpecLinear, self).__init__()
        self.IN = configs.instance_norm
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.individual = configs.individual
        self.channels = configs.enc_in

        self.length_ratio=(self.seq_len + self.pred_len)/self.seq_len

        if self.individual:
            self.freq_lin = nn.ModuleList()
            for i in range(self.channels):
                self.freq_lin.append(nn.Linear(self.seq_len//2+1, (self.seq_len + self.pred_len)//2+1, bias=False).to(torch.cfloat))
        else:
            self.freq_lin = nn.Linear(self.seq_len//2+1, (self.seq_len + self.pred_len)//2+1, bias=False).to(torch.cfloat) # complex layer for frequency upcampling]


    def forward(self, x, *args):
        if self.IN:
            x_mean = torch.mean(x, dim=1, keepdim=True)
            x = x - x_mean

        low_specx = torch.fft.rfft(x, dim=1)
        # print(low_specx.permute(0,2,1))
        if self.individual:
            low_specy = torch.zeros(
                [low_specx.size(0), (self.seq_len + self.pred_len)//2+1, low_specx.size(2)],
                dtype=low_specx.dtype
            ).to(low_specx.device)
            for i in range(self.channels):
                low_specy[:,:,i]=self.freq_lin[i](low_specx[:,:,i].permute(0,1)).permute(0,1)
        else:
            low_specy = self.freq_lin(low_specx.permute(0,2,1)).permute(0,2,1)
        y = torch.fft.irfft(low_specy, dim=1)
        y = y * self.length_ratio # energy compemsation for the length change
        
        if self.IN:
            y = y + x_mean
        return y[:,-self.pred_len:,:]



class RootPurgeModel(nn.Module):
    def __init__(self, configs, Model):
        super(RootPurgeModel, self).__init__()

        self.IN = configs.instance_norm
        self.lambda_ = configs.regu_coef
        self.order = configs.purge_order
    
        assert configs.residual_gt in ['zero', 'model_zero']
        self.residual_gt = configs.residual_gt
        self.padding_compensator = max(configs.seq_len/configs.pred_len,1.0)
        
        # set it to false so that base_model does not redo it
        configs_base_model = deepcopy(configs)
        configs_base_model.instance_norm = False
        self.base_model = Model(configs_base_model)


    @staticmethod
    def sample_residual(x):
        B, C, T, nS = x.shape
        indices = torch.randint(0, nS, (B, C), device=x.device)
        indices = indices.view(B, C, 1, 1).expand(-1, -1, T, -1)
        x_sampled = torch.gather(x, dim=3, index=indices)
        return x_sampled.squeeze(-1)

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        return self.base_model.load_state_dict(state_dict, strict, assign)

    def state_dict(self):
        return self.base_model.state_dict()

    def forward(self, *args):
        if self.training:
            return self.forward_train(*args)
        else:
            return self.forward_test(*args)

    def parameters(self, recurse: bool = True):
        return self.base_model.parameters(recurse)

    def forward_test(self, x, *args):
        if self.IN:
            x_mean = x.mean(dim = 1, keepdim = True)
            x = (x-x_mean)
        
        y = self.base_model(x)
        
        if self.IN:
            y = y + x_mean
        return y

    def forward_train(self, x, y, *args):
        if self.IN:
            x_mean = x.mean(dim = 1, keepdim = True)
            x = (x-x_mean)
            y = (y-x_mean)
        B, Tx, C = x.shape
        _, Ty, _ = y.shape
        # forward data
        yf_hat = self.base_model(x)
        residual_ = (y-yf_hat).detach() # B, Ty, C

        for i in range(self.order):
            if Tx == Ty:
                pass
            elif Tx > Ty:
                residual_ = F.pad(residual_.permute(0,2,1),  pad=(Tx-Ty, Tx-Ty)).unfold(-1, Tx, 1).permute(0,1,3,2) # B,C,nS,Tx -> B,C,Tx,nS
                residual_ = self.sample_residual(residual_).permute(0, 2, 1) # B, C, Tx -> B, Tx, C
            elif Tx < Ty:
                residual_ = residual_[:,:Tx,:]
            
            residual_ = self.base_model(residual_)

        val_y = residual_
        if self.residual_gt == "zero":
            val_gt = torch.zeros_like(val_y)
        elif self.residual_gt == "model_zero":
            val_gt = self.base_model(torch.zeros_like(x)).detach()
        floss = F.mse_loss(yf_hat, y)
        val_floss = F.mse_loss(val_y,val_gt)

        loss_ = floss+(val_floss * self.lambda_ * self.padding_compensator**self.order)
        return {"loss":loss_, "mse":floss, "checker":val_floss}



def adjust_learning_rate(optimizer, epoch, step, args, printout=False, scheduler=None):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    elif args.lradj == "type4":
        lr_adjust = {epoch: args.learning_rate if step < 3000 else args.learning_rate * (0.9 ** ((step - 2000) // 1000))}
    elif args.lradj == 'constant':
        lr_adjust = {epoch: args.learning_rate}
    elif args.lradj == '3':
        lr_adjust = {epoch: args.learning_rate if epoch < 10 else args.learning_rate*0.1}
    elif args.lradj == '4':
        lr_adjust = {epoch: args.learning_rate if epoch < 15 else args.learning_rate*0.1}
    elif args.lradj == '5':
        lr_adjust = {epoch: args.learning_rate if epoch < 25 else args.learning_rate*0.1}
    elif args.lradj == '6':
        lr_adjust = {epoch: args.learning_rate if epoch < 5 else args.learning_rate*0.1}  
    elif args.lradj == 'TST':
        assert scheduler is not None
        lr_adjust = {epoch: scheduler.get_last_lr()[0]}
    
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if printout: print('Updating learning rate to {}'.format(lr))


def vali(model, vali_loader, criterion, device_, args):
    total_loss_criterion = []
    total_loss_mae = []
    model.eval()
    #criterion = criterion.cpu()
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
            batch_x = batch_x.float().to(device_)
            batch_y = batch_y.float().to(device_)
            batch_x_mark = batch_x_mark.float().to(device_)
            batch_y_mark = batch_y_mark.float().to(device_)

            L = batch_y.shape[1]

            outputs = model(batch_x, batch_x_mark, None, None)

            outputs = outputs[:, -L:, :].detach()
            batch_y = batch_y.to(device_).detach()
            
            loss = criterion(outputs, batch_y)
            total_loss_criterion.append(loss.item())
            total_loss_mae.append(torch.abs(batch_y - outputs).mean().item())

    total_loss_criterion = np.average(total_loss_criterion)
    total_loss_mae = np.average(total_loss_mae)

    if hasattr(model, "ch_weight"):
        print(model.ch_weight)

    model.train()
    return total_loss_criterion, total_loss_mae

def test(model, test_loader, criterion, device_, args):
    all_loss_criterion = []
    batch_length = []
    all_loss_mae = []
    model.eval()
    #criterion = criterion.cpu()
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            batch_x = batch_x.float().to(device_)
            batch_y = batch_y.float().to(device_)
            batch_x_mark = batch_x_mark.float().to(device_)
            batch_y_mark = batch_y_mark.float().to(device_)

            L = batch_y.shape[1]

            outputs = model(batch_x, batch_x_mark, None, None)

            outputs = outputs[:, -L:, :].detach()
            batch_y = batch_y.to(device_).detach()
            
            loss = criterion(outputs, batch_y)
            batch_length.append(batch_x.shape[0])
            all_loss_criterion.append(loss.item())
            all_loss_mae.append(torch.abs(batch_y - outputs).mean().item())

    batch_weight = np.array(batch_length, np.float32)/np.sum(batch_length)
    total_loss_criterion = np.sum(np.array(all_loss_criterion)*batch_weight)
    total_loss_mae = np.sum(np.array(all_loss_mae)*batch_weight)

    if hasattr(model, "ch_weight"):
        print(model.ch_weight)

    model.train()
    return total_loss_criterion, total_loss_mae

def train(model, model_optim, criterion, 
          args, base_path, setting, 
          train_loader, vali_loader, test_loader, device_, val_every=50):
    path = os.path.join(base_path, setting, args.checkpoints)
    path_res_ = os.path.join(base_path, setting, "results")
    if not os.path.exists(path):
        os.makedirs(path)
    if not os.path.exists(path_res_):
        os.makedirs(path_res_)

    early_stopping = EarlyStopping(patience=args.train_steps, verbose=True)

    cur_step = 0
    epoch = 0
    min_val_mse, min_val_mae = np.inf, np.inf
    min_test_mse, min_test_mae = np.inf, np.inf
    while cur_step < args.train_steps:
        train_loss = []
        loss_item = None
        model.train()
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
            model_optim.zero_grad()
            batch_x = batch_x.float().to(device_)
            batch_y = batch_y.float().to(device_)
            batch_x_mark = batch_x_mark.float().to(device_)
            batch_y_mark = batch_y_mark.float().to(device_)

            L = batch_y.shape[1]

            if args.output_pred:
                outputs = model(batch_x)
                outputs = outputs[:, -L:, :]
                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())
            else:
                loss_dict = model(batch_x, batch_y)
                loss = loss_dict['loss']
                
                if loss_item is None:
                    loss_item = tuple(loss_dict.keys())
                train_loss.append([loss_dict[k].item() for k in loss_item])
                


            loss.backward()
            model_optim.step()
            

            # stdout information
            if (cur_step+1) % val_every == 0:
                train_loss_ = np.average(train_loss, 0)
                val_mse, val_mae = vali(model, vali_loader, criterion, device_, args)
                test_mse, test_mae = test(model, test_loader, criterion, device_, args)

                for param_group in model_optim.param_groups:
                    cur_lr = param_group['lr']

                if val_mse <= min_val_mse:
                    min_val_mse, min_val_mae = val_mse, val_mae
                    min_test_mse, min_test_mae = test_mse, test_mae

                if isinstance(train_loss_, np.ndarray):
                    train_loss_str = " ".join(f"{item_name_}:{val:.4f}" for item_name_, val in zip(loss_item,train_loss_))
                    print(
                        f"Step: {cur_step+1}, lr: {cur_lr:.5e} | Train {train_loss_str} | Vali loss: {val_mse:.3f}/{val_mae:.3f} | Test loss: {test_mse:.3f}/{test_mae:.3f}"
                    )
                else:
                    print(
                        f"Step: {cur_step+1}, lr: {cur_lr:.5e} | Train Loss: {train_loss_:.3f} | Vali loss: {val_mse:.3f}/{val_mae:.3f} | Test loss: {test_mse:.3f}/{test_mae:.3f}"
                    )
                early_stopping(val_mse, model, path)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break

            cur_step += 1
            if cur_step >= args.train_steps:
                break
        
        epoch += 1
        adjust_learning_rate(model_optim, epoch, cur_step, args)

    best_model_path = path + '/' + 'checkpoint.pth'
    model.load_state_dict(torch.load(best_model_path))

    result_file = os.path.join(path_res_, "forecast.txt")
    f = open(result_file, 'a')
    f.write(setting + "  \n")
    f.write(f'mse:{min_test_mse:.5f}, mae:{min_test_mae}')
    f.write('\n')
    f.write('\n')
    f.close()

    return model



parser = get_parser(description="Run forecasting model")


# data loader
parser.add_argument('--data', type=str, help='dataset type')
parser.add_argument('--l1_decay', type=float, default=0)
parser.add_argument('--l2_decay', type=float, default=0)
parser.add_argument('--train_steps', type = int, default=5000)
parser.add_argument("--instance_norm", type = str2bool, default=True)
parser.add_argument("--individual", type = str2bool, default=False)
parser.add_argument("--regu_coef", type=float, default=0.3)
parser.add_argument("--val_type", type=str, default="root") 
parser.add_argument("--residual_gt", type=str, default="zero") # zero or model_zero
parser.add_argument("--data_dsample", type=int, default=0)

parser.add_argument("--data_ending", type=int, default=400)
parser.add_argument("--data_noise", type=float, default=0.5)

parser.add_argument("--purge_order", type=int, default=1)

args = parser.parse_args()

base_path_ = "./model_result"
inst_norm_flag = "_IN" if args.instance_norm else ""

if args.model.lower() == "baselinearmodel":
    _model = BaseLinearModel
    args.mode = "sup" # force linear model to run with sup(ervise) mode
elif args.model.lower() == "speclinear":
    _model = SpecLinear
    args.mode = "sup" # force linear model to run with sup(ervise) mode
else:
    raise NotImplementedError("Unsupported variant")

if args.val_type.lower() == "root":
    _wrapper = RootPurgeModel
    args.output_pred = False
else:
    raise NotImplementedError("Unsupported variant")


if args.data.lower() == "synthetic":
    args.data_id = None
    args.enc_in = 1

    # length/noise
    def simulation_func_n(x, sigma):
        return np.sin(2*x) + np.cos(5*x) + 0.5*x + np.random.normal(loc=0.0, scale=1.0, size=x.shape)*sigma


    sim_func = partial(simulation_func_n, sigma=args.data_noise)
    data_core = Dataset_Function_MC_Core(
        functions = [sim_func], random_generator=None, 
        size = [args.seq_len, 0, args.pred_len], 
        scale=False, x_end=args.data_ending
    )


    train_dataset = Dataset_Function_MC(data_core, flag="train")
    val_dataset = Dataset_Function_MC(data_core, flag="val")
    test_dataset = Dataset_Function_MC(data_core, flag="test")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False
    )
    
    if args.purge_order == 1:
        purge_order_marker = ""
    else:
        purge_order_marker = f"_O{args.purge_order}"

    setting =  "{}_data_synth_T{}_std{}_d{:.2f}{}_dm{}_df{}_L{}_F{}_regC{:.4f}{}".format(
        args.model,
        args.data_ending,
        args.data_noise,
        args.dropout,
        inst_norm_flag,
        args.d_model,
        args.d_ff,
        args.seq_len,
        args.pred_len,
        args.regu_coef,
        purge_order_marker
    )

else:
    args.data_id = None
    train_dataset, train_loader = tslib_loader.data_provider(args = args, flag = "train")
    test_dataset, test_loader = tslib_loader.data_provider(args = args, flag = "test")
    val_dataset, val_loader = tslib_loader.data_provider(args = args, flag="val")

    if args.purge_order == 1:
        purge_order_marker = ""
    else:
        purge_order_marker = f"_O{args.purge_order}"


    #dataset_name = os.path.basename(os.path.normpath(args.root_path))
    dataset_name = args.model_id
    setting = "{}_data_{}_d{:.2f}{}_dm{}_df{}_L{}_F{}_regC{:.4f}{}".format(
        args.model,
        dataset_name,
        args.dropout,
        inst_norm_flag,
        args.d_model,
        args.d_ff,
        args.seq_len,
        args.pred_len,
        args.regu_coef,
        purge_order_marker
    )
print(setting)


fix_seed = args.seed
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)


device_ = "cuda" if torch.cuda.is_available() else "cpu"
print(f"using {device_} as training device!")


print("================start_train================")

base_model = _wrapper(args, _model)
model_ = base_model.to(device_)
model_criterion = nn.MSELoss()
model_optimizer = WeightDecayOptimizer(model_.parameters(), torch.optim.Adam,
                                    l1_decay=args.l1_decay, l2_decay=args.l2_decay,
                                    lr=args.learning_rate)

train(model_, model_optimizer, model_criterion, 
    args, base_path_, setting, 
    train_loader, val_loader, test_loader, 
    device_, val_every=50)

print("================end_train================")