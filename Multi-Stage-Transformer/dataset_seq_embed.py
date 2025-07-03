from torch.utils.data import TensorDataset, Dataset 

# padding to max_length
def ConfLog_Collate(batch, max_seq_length, pad_token):
    batched = [] 
    for sample in batch:
        if len(sample) < max_seq_length:
            sample.append(pad_token)
        batched.append(sample)
    return batched

class ConfLogDataset(TensorDataset):
    def __init__(self, log, conf):
        # list of log and corresponding configuration type 
        self.log = log 
        self.conf = conf 
    
    def __len__(self):
        return len(self.log)

    def __getitem__(self, idx):
        log_item, conf_item = self.log[idx], self.conf[idx]
        return log_item, conf_item 

class LogSequenceDataset(Dataset):
    
    # log_sequence(list of list): sequence of log embedding
    # conf(list): configuration of each sequence of log embedding 
    def __init__(self, log_sequence, padded_index, conf):
        self.log_sequence = log_sequence 
        self.padded_index = padded_index 
        self.conf = conf 

    def __len__(self):
        return len(self.log_sequence)

    def __getitem__(self, idx):
        log_seq, pad_index, conf_item = self.log_sequence[idx], self.padded_index[idx], self.conf[idx]
        return log_seq, pad_index, conf_item 