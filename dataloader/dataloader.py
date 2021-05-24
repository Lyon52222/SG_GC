from torch.utils.data import Dataset, DataLoader, TensorDataset
import json
import random
import torch
from torch import nn
import torch.utils.data as data

class SGDataLoader(Dataset):

    def get_classes(self):
        return self.ind_to_classes

    def get_predicates(self):
        return self.ind_to_predicates

    def __init__(self, opt):
        self.opt = opt
        self.batch_size = opt.batch_size
        self.box_topk = opt.box_topk
        self.rel_topk = opt.rel_topk
        self.filter_method = opt.filter_method
        self.threshold = opt.threshold

        self.used_nums = 0
        # load the json file which contains additional information about the dataset
        print('DataLoader loading data_info from: ', opt.custom_data_info_json)
        self.custom_data_info = json.load(open(opt.custom_data_info_json))
        print('DataLoader loading prediction_info from: ',
              opt.custom_prediction_json)
        # self.custom_prediction = json.load(open(opt.custom_prediction_json))
        if opt.filter_method == 'topk':
            self.processed_custom_prediction = json.load(open(opt.topk_custom_prediction_json))
        elif opt.filter_method == 'center':
            self.processed_custom_prediction = json.load(open(opt.center_custom_prediction_json))
            
        self.ind_to_classes = self.custom_data_info['ind_to_classes']
        self.ind_to_predicates = self.custom_data_info['ind_to_predicates']

        self.classes_size = len(self.ind_to_classes)
        self.rels_size = len(self.ind_to_predicates)
        print('classes size is ', self.classes_size)
        print('rels vocab size is ', self.rels_size)

        self._prefetch_process = BlobFetcher(self)
            # Terminate the child process when the parent exists


    def get_batch(self):
        data = self._prefetch_process.get()  # call one time to get a whole batch instead of fetching one by one instance
        return data 


    def __getitem__(self, index):
        box_labels = torch.tensor(self.processed_custom_prediction[index]['box_labels'],dtype=torch.long)
        box_features = torch.tensor(self.processed_custom_prediction[index]['box_features'],dtype=torch.float32)
        rel_labels = torch.tensor(self.processed_custom_prediction[index]['rel_labels'],dtype=torch.long)
        rels = torch.tensor(self.processed_custom_prediction[index]['rels'],dtype=torch.long)
        return box_labels, box_features, rel_labels, rels

    def __len__(self):
        return len(self.processed_custom_prediction)

    def get_dataloader(self):
        return DataLoader(dataset=self, batch_size=self.opt.batch_size, shuffle=True)


class BlobFetcher():
    """Experimental class for prefetching blobs in a separate process."""
    def __init__(self, dataloader, if_shuffle=False):
        """
        db is a list of tuples containing: imcrop_name, caption, bbox_feat of gt box, imname
        """
        self.dataloader = dataloader
        self.if_shuffle = if_shuffle
        self.batch_size = dataloader.batch_size

    # Add more in the queue
    def reset(self):
        """
        Two cases for this function to be triggered:
        1. not hasattr(self, 'split_loader'): Resume from previous training. Create the dataset given the saved split_ix and iterator
        2. wrapped: a new epoch, the split_ix and iterator have been updated in the get_minibatch_inds already.
        """
        # batch_size is 1, the merge is done in DataLoader class
        self.split_loader = iter(data.DataLoader(dataset=self.dataloader,
                                            batch_size=self.batch_size,  # should same as the number in ri_next = ri + self.batch_size
                                            shuffle=False,
                                            ))

    def _get_next_minibatch_inds(self):
        max_index = len(self.dataloader.processed_custom_prediction)
        wrapped = False
        last_batch = False
        ri = self.dataloader.used_nums  # count of images
        
        ri_next = ri + self.batch_size # should same as the number in "batch_size=self.batch_size,"
        if ri_next >= max_index:
            ri_next = 0
            if self.if_shuffle:
                random.shuffle(self.dataloader.processed_custom_prediction)
            wrapped = True
        
        self.dataloader.used_nums = ri_next  # shadow #data loaded by the dataloader 
        
        if wrapped is False and ri_next + self.batch_size >= max_index: # the next wrapped will be True, then current batch becomes last batch to be used
            last_batch = True

        return ri_next, wrapped, last_batch #ix, wrapped
    
    def get(self):
        if not hasattr(self, 'split_loader'):
            self.reset()
        
        ix, wrapped, last_batch = self._get_next_minibatch_inds()
        
        if wrapped:  # drop the final incomplete batch
            self.reset()  # self.dataloader.iterators[self.split] has been reset to 0 before call self.reset(); enter the new epoch
            ix, wrapped, last_batch = self._get_next_minibatch_inds()  # shadow #data loaded by the dataloader 
            tmp = self.split_loader.next()
        else:
            tmp = self.split_loader.next()  # shadow #data loaded by the dataloader

        #assert tmp[-1][2] == ix, "ix not equal"
        # return to self._prefetch_process[split].get() in Dataloader.get_batch()

        if last_batch:  # last batch
            wrapped = True
        return tmp 
