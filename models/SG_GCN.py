from utils.vectors import obj_edge_vectors
from torch._C import import_ir_module_from_buffer
from torch.nn import Module
from torch import nn
import models.lib.sg_gcn_backbone as GBackbone
import opts


class SG_GCN(Module):
    
    def __init__(self, opt, object_names, predicate_names):
        super(SG_GCN, self).__init__()

        self.object_names = object_names
        self.predicate_names = predicate_names
        self.embed_dim = opt.embed_dim
        self.sg_obj_cnt = len(self.object_names)
        self.sg_pred_cnt = len(self.predicate_names)
        self.GCN_layers = opt.gcn_layers
        self.GCN_dim = opt.gcn_dim
        self.GCN_residual = opt.gcn_residual
        self.GCN_use_bn = opt.gcn_bn
        self.object_feature_size = opt.object_feature_size

        self.obj_v_proj = nn.Linear(self.object_feature_size, self.GCN_dim)
        self.relu = nn.ReLU(inplace=True)

        embed_vecs = obj_edge_vectors(list(object_names), wv_dim=self.embed_dim)
        self.sg_obj_embed = nn.Embedding(self.sg_obj_cnt, self.embed_dim)
        self.sg_obj_embed.weight.data = embed_vecs.clone()
        self.obj_emb_proj = nn.Linear(self.embed_dim, self.GCN_dim)
        
        self.sg_pred_embed = nn.Embedding(self.sg_pred_cnt, self.embed_dim)
        p_embed_vecs = obj_edge_vectors(list(predicate_names), wv_dim=self.embed_dim)
        self.sg_pred_embed.weight.data = p_embed_vecs.clone()
        self.pred_emb_prj = nn.Linear(self.embed_dim, self.GCN_dim)

        self.gcn_backbone = GBackbone.gcn_backbone(GCN_layers=self.GCN_layers, GCN_dim=self.GCN_dim, GCN_residual=self.GCN_residual, GCN_use_bn=self.GCN_use_bn)

    def feat_fusion(self, obj_dist,object_feature, pred_dist):
        
        print(object_feature.dtype)
        #这里的obj_dist 对应为(b*N*sg_obj_cnt) 每个值对应N 属于那种obj的概率。
        #att_feats = self.obj_emb_proj(self.sg_obj_embed(obj_dist.view(-1, self.sg_obj_cnt)[:,1:].max(1)[1] + 1)).view(obj_dist.size(0), obj_dist.size(1), self.GCN_dim)

        # obj_emb = self.obj_emb_proj(self.sg_obj_embed(obj_dist.view(-1, self.sg_obj_cnt)[:,1:].max(1)[1] + 1)).view(obj_dist.size(0), obj_dist.size(1), self.GCN_dim)
        obj_emb = self.obj_emb_proj(self.sg_obj_embed(obj_dist.view(-1, 1))).view(obj_dist.size(0), obj_dist.size(1), self.GCN_dim)    
        object_feature = self.obj_v_proj(object_feature)
        object_feature = self.relu(object_feature + obj_emb)

        # att_feats = self.obj_emb_proj(self.sg_obj_embed(obj_dist.view(-1, 1))).view(obj_dist.size(0), obj_dist.size(1), self.GCN_dim)    
        #pred_fmap = self.pred_emb_prj(self.sg_pred_embed(pred_dist.view(-1, self.sg_pred_cnt)[:,1:].max(1)[1] + 1)).view(pred_dist.size(0), pred_dist.size(1), self.GCN_dim) 
        pred_fmap = self.pred_emb_prj(self.sg_pred_embed(pred_dist.view(-1, 1))).view(pred_dist.size(0), pred_dist.size(1), self.GCN_dim) 

        return object_feature, pred_fmap

    # def feat_fusion(self, obj_dist, att_feats, pred_dist):
        # """
        # Fuse visual and word embedding features for nodes and edges
        # """
        # # fuse features (visual, embedding) for each node in graph
        # if self.noun_fuse: # Sub-GC
            # obj_emb = self.obj_emb_proj(self.sg_obj_embed(obj_dist.view(-1, self.sg_obj_cnt)[:,1:].max(1)[1] + 1)).view(obj_dist.size(0), obj_dist.size(1), self.GCN_dim)
            # att_feats = self.obj_v_proj(att_feats)
            # att_feats = self.relu(att_feats + obj_emb)
        # else: # GCN-LSTM baseline that use full graph
            # att_feats = self.obj_v_proj(att_feats)

        # if self.pred_emb_type == 1: # hard emb, not including background
            # pred_emb = self.sg_pred_embed(pred_dist.view(-1, self.sg_pred_cnt)[:,1:].max(1)[1] + 1)
        # elif self.pred_emb_type == 2: # hard emb, including background
            # pred_emb = self.sg_pred_embed(pred_dist.view(-1, self.sg_pred_cnt).max(1)[1])
        # pred_fmap = self.pred_emb_prj(pred_emb).view(pred_dist.size(0), pred_dist.size(1), self.GCN_dim)
        # return att_feats, pred_fmap

    def forward(self, obj_dist=None,object_feature=None, rel_ind=None, pred_dist=None):
        
        # fuse features (visual, embedding) for each node in graph
        att_feats, pred_fmap = self.feat_fusion(obj_dist,object_feature, pred_dist)
        b = att_feats.size(0); N = att_feats.size(1); K = rel_ind.size(1); L = self.GCN_dim

        # GCN backbone (will expand feats to 5 counterparts)
        att_feats, x_pred = self.gcn_backbone(b,N,K,L,att_feats, obj_dist, pred_fmap, rel_ind)

        return att_feats, x_pred
