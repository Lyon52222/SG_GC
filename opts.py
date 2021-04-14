import argparse

def parse_opt():
    parser = argparse.ArgumentParser()
    ####### Original hyper-parameters #######
    # Data input settings
    parser.add_argument('--custom_data_info_json', type=str, default='data/custom_data_info.json',
                    help='path to the json file containing data infos')

    parser.add_argument('--custom_prediction_json', type=str, default='data/custom_prediction.json',
                    help='path to the json file containing prediction infos')

    parser.add_argument('--box_topk', type=int, default=5,
                    help='the topk boxs you will select')
    
    parser.add_argument('--rel_topk', type=int, default=10,
                    help='the topk rels you will select')
    
    parser.add_argument('--start_from', type=str, default=None,
                    help="""continue training from saved model at this path. Path must contain files saved by previous training process: 
                        'infos.pkl'         : configuration;
                        'checkpoint'        : paths to model file(s) (created by tf).
                                              Note: this file contains absolute paths, be careful when moving files around;
                        'model.ckpt-*'      : file(s) with model definition (created by tf)
                    """)
    parser.add_argument('--cached_tokens', type=str, default='coco-train-idxs',
                    help='Cached token file for calculating cider score during self critical training.')

    # Model settings
    parser.add_argument('--caption_model', type=str, default="show_tell",
                    help='show_tell, show_attend_tell, all_img, fc, att2in, att2in2, att2all2, adaatt, adaattmo, topdown, stackatt, denseatt, transformer')
    parser.add_argument('--rnn_size', type=int, default=512,
                    help='size of the rnn in number of hidden nodes in each layer')
    parser.add_argument('--num_layers', type=int, default=1,
                    help='number of layers in the RNN')
    parser.add_argument('--rnn_type', type=str, default='lstm',
                    help='rnn, gru, or lstm')
    parser.add_argument('--input_encoding_size', type=int, default=512,
                    help='the encoding size of each token in the vocabulary, and the image.')
    parser.add_argument('--att_hid_size', type=int, default=512,
                    help='the hidden size of the attention MLP; only useful in show_attend_tell; 0 if not using hidden layer')
    parser.add_argument('--fc_feat_size', type=int, default=2048,
                    help='2048 for resnet, 4096 for vgg')
    parser.add_argument('--att_feat_size', type=int, default=2048,
                    help='2048 for resnet, 512 for vgg')
    parser.add_argument('--logit_layers', type=int, default=1,
                    help='number of layers in the RNN')
    parser.add_argument('--use_bn', type=int, default=0,
                    help='If 1, then do batch_normalization first in att_embed, if 2 then do bn both in the beginning and the end of att_embed')


    # Optimization: General
    parser.add_argument('--max_epochs', type=int, default=-1,
                    help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=2,
                    help='minibatch size')
    parser.add_argument('--grad_clip', type=float, default=0.1, #5.,
                    help='clip gradients at this value')
    parser.add_argument('--drop_prob_lm', type=float, default=0.5,
                    help='strength of dropout in the Language Model RNN')
    parser.add_argument('--self_critical_after', type=int, default=-1,
                    help='After what epoch do we start finetuning the CNN? (-1 = disable; never finetune, 0 = finetune from start)')
    parser.add_argument('--seq_per_img', type=int, default=5,
                    help='number of captions to sample for each image during training. Done for efficiency since CNN forward pass is expensive. E.g. coco has 5 sents/image')

 
    ####### Graph captioning model hyper-parameters #######
   
    parser.add_argument('--embed_dim', type=int, default=300, 
                    help='dim of word embeddings')
    parser.add_argument('--gcn_dim', type=int, default=1024, 
                    help='dim of the node/edge features in GCN')
    parser.add_argument('--pred_emb_type', type=int, default=1, 
                    help='predicate embedding type')
    parser.add_argument('--gcn_layers', type=int, default=2, 
                    help='the layer number of GCN')
    parser.add_argument('--gcn_residual', type=int, default=2,
                    help='2: there is a skip connection every 2 GCN layers')
    parser.add_argument('--gcn_bn', type=int, default=0, 
                    help='0: not use BN in GCN layers')
    parser.add_argument('--sampling_prob', type=float, default=0.0, 
                    help='Schedule sampling probability')
   

    

    args = parser.parse_args()


    return args