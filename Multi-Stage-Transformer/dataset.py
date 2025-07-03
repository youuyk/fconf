from torch.utils.data import TensorDataset 

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

