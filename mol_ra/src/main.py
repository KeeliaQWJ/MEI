import os
import argparse
import data_processing
import train
from property_pred import pp_data_processing, pp_train_lr
from ged_pred import gp_data_processing, gp_train
#from visualization import visualize


def print_setting(args):
    print('\n===========================')
    for k, v, in args.__dict__.items():
        print('%s: %s' % (k, v))
    print('===========================\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='the index of gpu device')

    #'''
    # pretraining / chemical reaction prediction
    parser.add_argument('--task', type=str, default='pretrain', help='downstream task')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--dataset', type=str, default='esp', help='dataset name')
    parser.add_argument('--epoch', type=int, default=40, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--gnn', type=str, default='tag', help='name of the GNN model')
    parser.add_argument('--layer', type=int, default=2, help='number of GNN layers')
    parser.add_argument('--dim', type=int, default=1024, help='dimension of molecule embeddings')
    parser.add_argument('--margin', type=float, default=4.0, help='margin in contrastive loss')
    parser.add_argument('--lr', type=float, default=2e-5, help='learning rate')
    parser.add_argument('--save_model', type=bool, default=True, help='save the trained model to disk')
    #'''

    '''
    # molecule property prediction
    parser.add_argument('--task', type=str, default='property_pred', help='downstream task')
    parser.add_argument('--pretrained_model', type=str, default='gcn_1024', help='the pretrained model')
    parser.add_argument('--dataset', type=str, default='esp', help='dataset name')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size for calling the pretrained model')
    '''

    args = parser.parse_args()
    dataset = args.dataset
    #mode = args.mode
    print_setting(args)
    print('current working directory: ' + os.getcwd() + '\n')

    if args.task == 'pretrain':
        data = data_processing.load_data(args)
        #data = data_processing.preprocess(dataset = args.dataset)
        train.train(args, data)
    elif args.task == 'property_pred':
        data = pp_data_processing.load_data(args)
        pp_train_lr.train(args, data)
    else:
        raise ValueError('unknown task')


if __name__ == '__main__':
    main()
