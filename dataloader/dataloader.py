from torch.utils.data import Dataset, DataLoader, TensorDataset
import json
import random
import torch
from torch.nn.utils.rnn import pad_sequence
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


    def __getitem__(self, index):
        box_labels = torch.tensor(self.processed_custom_prediction[index]['box_labels'],dtype=torch.long)
        box_features = torch.tensor(self.processed_custom_prediction[index]['box_features'],dtype=torch.float32)
        rel_labels = torch.tensor(self.processed_custom_prediction[index]['rel_labels'],dtype=torch.long)
        rels = torch.tensor(self.processed_custom_prediction[index]['rels'],dtype=torch.long)
        return box_labels, box_features, rel_labels, rels

    def __len__(self):
        return len(self.processed_custom_prediction)

    def get_dataloader(self):
        return DataLoader(dataset=self, batch_size=self.opt.batch_size, shuffle=True, collate_fn=self.collate_func)


    def collate_func(self,batch_dic):
        print(len(batch_dic))
        # batch_len = len(batch_dic)
        # max_box_label = max([len(dic['box_labels']) for dic in batch_dic])
        # max_rel_label = max([len(dic['rel_labels']) for dic in batch_dic])
        box_labels = []
        box_features = []
        rel_labels = []
        rels = []
        for i in range(len(batch_dic)):
            dic = batch_dic[i]
            box_labels.append(dic[0])
            box_features.append(dic[1])
            rel_labels.append(dic[2])
            rels.append(dic[3])
        res = []
        res.append(pad_sequence(box_labels, batch_first=True))
        res.append(pad_sequence(box_features, batch_first=True))
        res.append(pad_sequence(rel_labels, batch_first=True))
        res.append(pad_sequence(rels, batch_first=True))
        return res



