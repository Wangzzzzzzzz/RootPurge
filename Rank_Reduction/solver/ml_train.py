from copy import deepcopy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np





def slice_target_dataset(dataset, index, offset = 1):
    buffer = []
    used_index = []
    for idx in index:
        if idx+offset < len(dataset):
            item = dataset[idx+offset]
            buffer.append(item[0])
            used_index.append(idx)
    
    return torch.tensor(np.array(buffer)).float(), used_index


# Training function
def train(model, dataloader, dataset, criterion, optimizer, scheduler, device, temporal_consistency=0):
    model.train()
    running_loss = 0.0
    for item in dataloader:
        batch_x, batch_y = item[0],item[1]
        if len(item) == 5: 
            index = item[-1] 
        else:
            index = None

        if index is not None:
            temporal_offset_data, used_index = slice_target_dataset(dataset=dataset,
                                                                    index=index,
                                                                    offset=1)
            offset_data = temporal_offset_data.float().to(device)
            offset_output = model(offset_data)
        
        batch_x, batch_y = batch_x.float().to(device), batch_y.float().to(device)
        optimizer.zero_grad()
        outputs = model(batch_x)

        try:
            loss = criterion(outputs, batch_y, model)
        except:
            loss = criterion(outputs, batch_y)

        if temporal_consistency:
            if outputs.shape[0] == offset_output.shape[0]:
                consistency_loss = ( (outputs[:,1:,:] - offset_output[:,:-1,:])**2 ).mean()
                loss += temporal_consistency*consistency_loss
                #print(consistency_loss)


        loss.backward()
        optimizer.step()
        #optimizer.step(closure)
        if scheduler is not None:
            scheduler.step()
        
        # obtain loss
        with torch.no_grad():
            outputs = model(batch_x)
            try:
                loss = criterion(outputs, batch_y, model)
            except:
                loss = criterion(outputs, batch_y)
        running_loss += loss.item() * batch_x.size(0)
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss



# Training function
def train_constrained(
        model, constrains, dataloader, dataset, criterion, 
        optim_model, optim_constrain, scheduler, device,
        consistency_offset = 1):
    model.train()
    running_loss = 0.0
    for item in dataloader:

        #random_scale = np.random.uniform(-1, 1, 1)[0]
        #random_scale = 1
        batch_x, batch_y = item[0],item[1]
        if len(item) == 5: 
            index = item[-1] 
        else:
            index = None

        if index is not None:
            temporal_offset_data, used_index = slice_target_dataset(dataset=dataset,
                                                                    index=index,
                                                                    offset=consistency_offset)
            offset_data = temporal_offset_data.float().to(device)
            offset_output = model(offset_data)
        
        batch_x, batch_y = batch_x.float().to(device), batch_y.float().to(device)

        optim_model.zero_grad()
        optim_constrain.zero_grad()
        outputs = model(batch_x)

        try:
            loss = criterion(outputs, batch_y, model)
        except:
            loss = criterion(outputs, batch_y)

        if consistency_offset != 0  and outputs.shape[0] == offset_output.shape[0]:
            constrain_loss = constrains(outputs[:,consistency_offset:,:], offset_output[:,:-consistency_offset,:])
            loss += constrain_loss


        loss.backward()
        optim_model.step()
        optim_constrain.step()
        #optimizer.step(closure)
        if scheduler is not None:
            scheduler.step()
        
        # obtain loss
        with torch.no_grad():
            outputs = model(batch_x)
            try:
                loss = criterion(outputs, batch_y, model)
            except:
                loss = criterion(outputs, batch_y)
        running_loss += loss.item() * batch_x.size(0)
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss


# Validation function
def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for item in dataloader:
            batch_x, batch_y = item[0],item[1]
            batch_x, batch_y = batch_x.float().to(device), batch_y.float().to(device)
            
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            running_loss += loss.item() * batch_x.size(0)
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss

# Testing function
def test(model, test_dataloader, device):
    model.eval()
    preds = []
    gts = []

    errors = []
    with torch.no_grad():
        for item in test_dataloader:
            batch_x, batch_y = item[0],item[1]
            batch_x = batch_x.float().to(device)
            outputs = model(batch_x)

            pred_ = outputs.clone().detach().cpu().numpy()
            gt_ = batch_y.clone().detach().cpu().numpy()

            errors.append((pred_-gt_))
            # preds.append(pred_)
            # gts.append(gt_)

    # preds = np.concatenate(preds, axis = 0)
    # gts = np.concatenate(gts, axis = 0)
    errors = np.concatenate(errors, axis = 0)
    mse = (errors**2).mean()
    mae = np.abs(errors).mean()

    return mse, mae

# Putting it all together
def run_training(train_loader,
                 train_data,
                 val_loader, 
                 test_loader, 
                 model,
                 loss_fn=nn.MSELoss(),
                 num_epochs=20, 
                 opt = optim.AdamW,
                 learning_rate=0.001,
                 scheduler = None,
                 temporal_consistency=0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)
    criterion = loss_fn
    #optimizer = optim.LBFGS(model.parameters(), lr=learning_rate, max_iter=100, line_search_fn = "strong_wolfe", tolerance_grad=1e-10, tolerance_change=1e-11)
    optimizer = opt(model.parameters(), lr = learning_rate)

    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, train_data, criterion, optimizer, None, device, temporal_consistency)
        val_loss = validate(model, val_loader, nn.MSELoss(), device)
        test_result = test(model, test_loader, device)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, test_loss: {test_result[0]:.4f}")
        
        if val_loss < best_val_loss and epoch > 20:
            best_val_loss = val_loss
            best_model_wts = deepcopy(model.state_dict())

        if scheduler is not None and hasattr(scheduler, "step"):
            scheduler.step()

    model.load_state_dict(best_model_wts)
    test_result = test(model, test_loader, device)
    print(f"mse: {test_result[0]}, mae: {test_result[1]}")

    # if hasattr(model, "plot_heatmap"):
    #     model.plot_heatmap()
    return test_result[0], test_result[1]


def run_constrained_training(
        train_loader,
        train_data,
        val_loader, 
        test_loader, 
        model,
        model_constrain,
        loss_fn=nn.MSELoss(),
        num_epochs=20, 
        opt = optim.AdamW,
        learning_rate=0.001,
        scheduler = None,
        offset = 1
    ):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    model = model.to(device)
    model_constrain = model_constrain.to(device)
    criterion = loss_fn
    #optimizer = optim.LBFGS(model.parameters(), lr=learning_rate, max_iter=100, line_search_fn = "strong_wolfe", tolerance_grad=1e-10, tolerance_change=1e-11)
    optimizer = opt(model.parameters(), lr = learning_rate)
    optimizer_constrain = opt(model_constrain.parameters(), lr=learning_rate, maximize=True)

    if scheduler is not None:
        s1 = scheduler(optimizer, step_size=100)
        s2 = scheduler(optimizer_constrain, step_size=100)

    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        train_loss = train_constrained(
            model, model_constrain, 
            train_loader, train_data, 
            criterion, optimizer, optimizer_constrain,
            None, device, offset
        )
        val_loss = validate(model, val_loader, nn.MSELoss(), device)
        test_result = test(model, test_loader, device)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, test_loss: {test_result[0]:.4f}")
        
        if val_loss < best_val_loss and epoch > 20:
            best_val_loss = val_loss
            best_model_wts = deepcopy(model.state_dict())

        if scheduler is not None and hasattr(scheduler, "step"):
            s1.step()
            s2.step()

    model.load_state_dict(best_model_wts)
    test_result = test(model, test_loader, device)
    print(f"mse: {test_result[0]}, mae: {test_result[1]}")

    # if hasattr(model, "plot_heatmap"):
    #     model.plot_heatmap()
    return test_result[0], test_result[1]
