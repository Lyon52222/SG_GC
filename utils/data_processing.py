import json
from opts import parse_opt


def filter_topk(opt):
    custom_prediction = json.load(open(opt.custom_prediction))


if __name__ == '__main__':
    opt = parse_opt()
    filter_topk(opt)
