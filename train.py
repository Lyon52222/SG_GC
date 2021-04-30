from models.SG_GCN import SG_GCN
from opts import parse_opt
import torch
from dataloader.dataloader import SGDataLoader
if __name__ == '__main__':
    opt = parse_opt()

    loader = SGDataLoader(opt)
    train_loader = loader.get_dataloader()
    model = SG_GCN(opt, loader.get_classes(), loader.get_predicates())


    for batch_labels, batch_rel_labels, batch_rels in train_loader:
        print(batch_labels[0])
        #batch_size,object_nums
        print(batch_labels.shape)
        print('---')
        print(batch_rel_labels[0])
        #batch_size,rel_nums
        print(batch_rel_labels.shape)
        print('---')
        print(batch_rels[0])
        #batch_size,rel_nums,2
        print(batch_rels.shape)

        print('over')

        output =  model(batch_labels, batch_rels, batch_rel_labels)
        #batch_size,object_nums,embedding_size
        print(output[0].shape)
        #batch_size,rel_nums,embedding_size
        print(output[1].shape)

        print(output)

        break


    #output = model()
