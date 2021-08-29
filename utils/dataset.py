from torch.utils.data import Sampler, Dataset
from collections import OrderedDict
from random import shuffle
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence


class BucketDataset(Dataset):

    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        if self.targets is None:
            return self.inputs[index]
        else:
            return self.inputs[index], self.targets[index]


class BucketBatchSampler(Sampler):
    # want inputs to be an array
    def __init__(self, inputs, batch_size):
        self.batch_size = batch_size
        ind_n_len = []
        for i, p in enumerate(inputs):
            ind_n_len.append((i, p.shape[0]))
        self.ind_n_len = ind_n_len
        self.batch_list = self._generate_batch_map()
        self.num_batches = len(self.batch_list)

    def _generate_batch_map(self):
        # shuffle all of the indices first so they are put into buckets differently
        shuffle(self.ind_n_len)
        # Organize lengths, e.g., batch_map[10] = [30, 124, 203, ...] <= indices of sequences of length 10
        batch_map = OrderedDict()
        for idx, length in self.ind_n_len:
            if length not in batch_map:
                batch_map[length] = [idx]
            else:
                batch_map[length].append(idx)
        # Use batch_map to split indices into batches of equal size
        # e.g., for batch_size=3, batch_list = [[23,45,47], [49,50,62], [63,65,66], ...]
        batch_list = []
        for length, indices in batch_map.items():
            for group in [indices[i:(i + self.batch_size)] for i in range(0, len(indices), self.batch_size)]:
                batch_list.append(group)
        return batch_list

    def batch_count(self):
        return self.num_batches

    def __len__(self):
        return len(self.ind_n_len)

    def __iter__(self):
        self.batch_list = self._generate_batch_map()
        # shuffle all the batches so they arent ordered by bucket size
        shuffle(self.batch_list)
        for i in self.batch_list:
            yield i

def pad_packed_collate(batch):
    """Puts data, and lengths into a packed_padded_sequence then returns
       the packed_padded_sequence and the labels. Set use_lengths to True
       to use this collate function.
       Args:
         batch: (list of tuples) [(audio, target)].
             audio is a FloatTensor
             target is a LongTensor with a length of 8
       Output:
         packed_batch: (PackedSequence), see torch.nn.utils.rnn.pack_padded_sequence
         labels: (Tensor), labels from the file names of the wav.
    """

    if len(batch) == 1:
        sigs, labels = batch[0][0], batch[0][1]
        sigs = sigs.t()
        lengths = [sigs.size(0)]
        sigs.unsqueeze_(0)
        labels.unsqueeze_(0)
    if len(batch) > 1:
        sigs, labels, lengths = zip(*[(a.t(), b, a.size(1)) for (a,b) in sorted(batch, key=lambda x: x[0].size(1), reverse=True)])
        max_len, n_feats = sigs[0].size()
        sigs = [torch.cat((s, torch.zeros(max_len - s.size(0), n_feats)), 0) if s.size(0) != max_len else s for s in sigs]
        sigs = torch.stack(sigs, 0)
        labels = torch.stack(labels, 0)
    packed_batch = pack_padded_sequence(Variable(sigs), lengths, batch_first=True)
    return packed_batch, labels
