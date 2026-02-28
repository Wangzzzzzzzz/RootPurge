from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred
from torch.utils.data import DataLoader

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
}



def data_provider(data, 
                  batch_size, 
                  root_path, 
                  data_path, 
                  size,
                  flag,
                  embed = "timeF",
                  freq="h",
                  features="M", 
                  target="OT"
                  ):
    Data = data_dict[data]
    timeenc = 0 if embed != 'timeF' else 1

    seq_len, label_len, pred_len = size[0], size[1], size[2]

    shuffle_flag = False if flag == 'test' else True
    drop_last = False if (flag == "test" or flag == "val") else True
    batch_size = batch_size
    freq = freq

    data_set = Data(
        root_path=root_path,
        data_path=data_path,
        flag=flag,
        size=[seq_len, label_len, pred_len],
        features=features,
        target=target,
        timeenc=timeenc,
        freq=freq
    )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        drop_last=drop_last)
    return data_set, data_loader
