import json
import torch
from opts import parse_opt


def filter_topk(opt):
        print('DataLoader loading prediction_info from: ',opt.custom_prediction_json)
        custom_prediction = json.load(open(opt.custom_prediction_json))
        processed_custom_prediction = []
        for index in range(len(custom_prediction)):
            box_labels = custom_prediction[str(
                index)]['bbox_labels'][:opt.box_topk]
            box_features = custom_prediction[str(index)]['bbox_features'][:opt.box_topk]
            #box_scores = self.custom_prediction[str(index)]['bbox_scores'][:self.box_topk]
            all_rel_labels = custom_prediction[str(index)]['rel_labels']
            #all_rel_scores = self.custom_prediction[str(index)]['rel_scores']
            all_rel_pairs = custom_prediction[str(index)]['rel_pairs']

            rel_labels = []
            rels = []
            for i in range(len(all_rel_pairs)):
                if all_rel_pairs[i][0] < opt.box_topk and all_rel_pairs[i][1] < opt.box_topk:
                    rel_labels.append(all_rel_labels[i])
                    rels.append(all_rel_pairs[i])
            rel_labels = rel_labels[:opt.rel_topk]
            rels = rels[:opt.rel_topk]

            # box_labels = torch.tensor(box_labels, dtype=torch.long)
            # box_features = torch.tensor(box_features, dtype=torch.float32)
            # rel_labels = torch.tensor(rel_labels, dtype=torch.long)
            # rels = torch.tensor(rels, dtype=torch.long)
            processed_custom_prediction.append({"box_labels":box_labels,"box_features":box_features,"rel_labels":rel_labels,"rels":rels})
        with open(opt.topk_custom_prediction_json,'w') as topk_output:
            json.dump(processed_custom_prediction,topk_output)


def filter_center(opt):
    print('DataLoader loading prediction_info from: ',opt.custom_prediction_json)
    custom_prediction = json.load(open(opt.custom_prediction_json))
    processed_custom_prediction = []
    for index in range(len(custom_prediction)):
        box_labels = custom_prediction[str(index)]['bbox_labels']
        box_features = custom_prediction[str(index)]['bbox_features']
        all_rel_pairs = custom_prediction[str(index)]['rel_pairs']
        all_rel_labels = custom_prediction[str(index)]['rel_labels']
        all_rel_scores = custom_prediction[str(index)]['rel_scores']

        weights = [0]*len(box_labels)
        for i, pair in enumerate(all_rel_pairs):
            weights[pair[0]] = weights[pair[0]] + \
                (1 if all_rel_scores[i]
                 > opt.threshold else 0)
            weights[pair[1]] = weights[pair[1]] + \
                (1 if all_rel_scores[i]
                 > opt.threshold else 0)
        center = weights.index(max(weights))
        objects = []
        obj_features = []
        rel_labels = []
        rel_pairs = []
        for i, pair in enumerate(all_rel_pairs):
            if all_rel_scores[i] > opt.threshold and (pair[0] == center or pair[1] == center):
                if pair[0] not in objects:
                    objects.append(pair[0])
                    obj_features.append(box_features[pair[0]])
                if pair[1] not in objects:
                    objects.append(pair[1])
                    obj_features.append(box_features[pair[0]])
                rel_labels.append(all_rel_labels[i])
                rel_pairs.append(
                    [objects.index(pair[0]), objects.index(pair[1])])
        # box_labels = torch.tensor(objects, dtype=torch.long)
        # obj_features = torch.tensor(obj_features, dtype=torch.float32)
        # rel_labels = torch.tensor(rel_labels, dtype=torch.long)
        # rel_pairs = torch.tensor(rel_pairs, dtype=torch.long)
        processed_custom_prediction.append({"box_labels":objects,"box_features":obj_features,"rel_labels":rel_labels,"rels":rel_pairs})
    with open(opt.center_custom_prediction_json,'w') as center_output:
        json.dump(processed_custom_prediction,center_output)    
if __name__ == '__main__':
    opt = parse_opt()
    # filter_topk(opt)
    filter_center(opt)
