from models.SG_GCN import SG_GCN
from opts import parse_opt
import torch
from dataloader.dataloader import SGDataLoader
if __name__ == '__main__':
    opt = parse_opt()

    loader = SGDataLoader(opt)
    
    batch_labels, batch_features, batch_rel_labels, batch_rels = loader.get_batch() 
    print(batch_labels[0])
    #batch_size,object_nums
    print(batch_labels.shape)
    print('---')
    print(batch_features[0])
    #batch_size,object_nums,feature_size
    print(batch_features.shape)
    print('---')
    print(batch_rel_labels[0])
    #batch_size,rel_nums
    print(batch_rel_labels.shape)
    print('---')
    print(batch_rels[0])
    #batch_size,rel_nums,2
    print(batch_rels.shape)

    print('over')
