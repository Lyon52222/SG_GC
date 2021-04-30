from torch.utils.data import Dataset, DataLoader, TensorDataset
import json
import torch


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
        # load the json file which contains additional information about the dataset
        print('DataLoader loading data_info from: ', opt.custom_data_info_json)
        self.custom_data_info = json.load(open(opt.custom_data_info_json))
        print('DataLoader loading prediction_info from: ',
              opt.custom_prediction_json)
        self.custom_prediction = json.load(open(opt.custom_prediction_json))
        self.ind_to_classes = self.custom_data_info['ind_to_classes']
        self.ind_to_predicates = self.custom_data_info['ind_to_predicates']

        self.classes_size = len(self.ind_to_classes)
        self.rels_size = len(self.ind_to_predicates)
        print('classes size is ', self.classes_size)
        print('rels vocab size is ', self.rels_size)

    def __getitem__(self, index):
        if self.filter_method == 'topk':
            #image_path = self.custom_data_info['idx_to_files'][index]
            #boxes = self.custom_prediction[str(index)]['bbox'][:self.box_topk]
            box_labels = self.custom_prediction[str(
                index)]['bbox_labels'][:self.box_topk]
            #box_scores = self.custom_prediction[str(index)]['bbox_scores'][:self.box_topk]
            all_rel_labels = self.custom_prediction[str(index)]['rel_labels']
            #all_rel_scores = self.custom_prediction[str(index)]['rel_scores']
            all_rel_pairs = self.custom_prediction[str(index)]['rel_pairs']

            rel_labels = []
            rels = []
            for i in range(len(all_rel_pairs)):
                if all_rel_pairs[i][0] < self.box_topk and all_rel_pairs[i][1] < self.box_topk:
                    rel_labels.append(all_rel_labels[i])
                    rels.append(all_rel_pairs[i])
            rel_labels = rel_labels[:self.rel_topk]
            rels = rels[:self.rel_topk]

            box_labels = torch.tensor(box_labels, dtype=torch.long)
            rel_labels = torch.tensor(rel_labels, dtype=torch.long)
            rels = torch.tensor(rels, dtype=torch.long)

            return box_labels, rel_labels, rels

        elif self.filter_method == 'center':
            box_labels = self.custom_prediction[str(index)]['bbox_labels']
            all_rel_pairs = self.custom_prediction[str(index)]['rel_pairs']
            all_rel_labels = self.custom_prediction[str(index)]['rel_labels']
            all_rel_scores = self.custom_prediction[str(index)]['rel_scores']

            weights = [0]*len(box_labels)
            for i, pair in enumerate(all_rel_pairs):
                weights[pair[0]] = weights[pair[0]] + \
                    (1 if all_rel_scores[i]
                     > self.threshold else 0)
                weights[pair[1]] = weights[pair[1]] + \
                    (1 if all_rel_scores[i]
                     > self.threshold else 0)

            center = weights.index(max(weights))

            objects = []
            rel_labels = []
            rel_pairs = []
            for i, pair in enumerate(all_rel_pairs):
                if all_rel_scores[i] > self.threshold and (pair[0] == center or pair[1] == center):
                    if pair[0] not in objects:
                        objects.append(pair[0])
                    if pair[1] not in objects:
                        objects.append(pair[1])

                    rel_labels.append(all_rel_labels[i])
                    rel_pairs.append(
                        [objects.index(pair[0]), objects.index(pair[1])])
            box_labels = torch.tensor(objects, dtype=torch.long)
            rel_labels = torch.tensor(rel_labels, dtype=torch.long)
            rel_pairs = torch.tensor(rel_pairs, dtype=torch.long)
            return box_labels, rel_labels, rel_pairs

    def __len__(self):
        return len(self.custom_data_info['idx_to_files'])

    def get_dataloader(self):
        return DataLoader(dataset=self, batch_size=self.opt.batch_size, shuffle=True)
