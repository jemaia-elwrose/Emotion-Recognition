import time
import pickle
import random
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from losses import MaskedNLLLoss
from model import DialogueGCN_DailyModel
from dataloader import DailyDialogueDataset
from helpers import time_since, save_model, load_model
from modules import LSTMModel, GRUModel, DialogRNNModel
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

# Using seed for reproduction of result
seed = 77777
def seed_everything(seed=seed):
    """
    Function to seed everything to produce same result everytime.

    Args:
        seed: seed integer, Default: 77777

    Returns:
        None
        
    
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_DailyDialogue_loader(path, batch_size=32, val_size=0.2,
                             shuffle=True, num_workers=0, pin_memory=False):
    """
    Function to load the DailyDialogue dataset.

    Args:
        path: path to the pickle file
        batch_size: batch_size, Default: 32
        val_size: validation size, Default: 0.2
        shuffle: shuffle data, Default, True
        num_workers: num workers, Default: 0
        pin_memory: pin memory, Default: False

    Returns:
        train_loader, valid_loader, test_loader
        
    """

    trainset = DailyDialogueDataset('train', path)
    validset = DailyDialogueDataset('valid', path)
    testset = DailyDialogueDataset('test', path)

    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              shuffle=shuffle,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    valid_loader = DataLoader(validset,
                              batch_size=batch_size,
                              shuffle=shuffle,
                              collate_fn=validset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=num_workers,
                             pin_memory=pin_memory)

    return train_loader, valid_loader, test_loader


def train_or_eval_graph_model(model, loss_function, dataloader, epoch,
                              cuda_flag, optimizer=None, train=False):
    """
    Function to train/eval model for graph model on dataset for 1 epoch.

    Args:
        model: model
        loss_function: loss function
        dataloader: dataloader
        epoch: current epoch
        cuda_flag: cuda device
        optimizer: optimizer
        train: True(training)/False(validation), Default: False

    Returns:
        avg_loss, avg_accuracy, labels, preds, avg_fscore,
        vids, ei, et, en, el, precision, recall
        
    """

    device = torch.device('cuda')

    losses, preds, labels = [], [], []

    assert not train or optimizer != None
    if train:
        model.train()
    else:
        model.eval()

    # seed_everything()
    for data in dataloader:
        if train:
            optimizer.zero_grad()

        textf, qmask, umask, label = [d.to(device) for d in data[:-1]] if cuda_flag else data[:-1]
        lengths = [(umask[j] == 1).nonzero().tolist()[-1][0] + 1 for j in range(len(umask))]

        log_prob, e_i, e_n, e_t, e_l = model(textf, qmask, umask, lengths)
        label = torch.cat([label[j][:lengths[j]] for j in range(len(label))])
        loss = loss_function(log_prob, label)

        preds.append(torch.argmax(log_prob, dim=1).cpu().numpy())
        labels.append(label.cpu().numpy())
        losses.append(loss.item())

        if train:
            loss.backward()

            if args.tensorboard:
                for param in model.named_parameters():
                    writer.add_histogram(param[0], param[1].grad, epoch)

            optimizer.step()

    if preds != []:
        preds = np.concatenate(preds)
        labels = np.concatenate(labels)
    else:
        return float('nan'), float('nan'), [], [], float('nan'), [], [], [], [], []

    labels = np.array(labels)
    preds = np.array(preds)

    avg_loss = round(np.sum(losses) / len(losses), 4)
    avg_accuracy = round(accuracy_score(labels, preds) * 100, 2)

    # Masking the `no_emotion` label = 3
    avg_fscore = round(f1_score(labels, preds, average='micro',
                                labels=[0,1,2,4,5,6]) * 100, 2)

    # Precision and Recall
    precision = round(precision_score(labels, preds, average='micro',
                                      labels=[0,1,2,4,5,6]) * 100, 2)
    recall = round(recall_score(labels, preds, average='micro',
                                labels=[0,1,2,4,5,6]) * 100, 2)

    return avg_loss, avg_accuracy, labels, preds, avg_fscore, precision, recall


def train_or_eval_model(model, loss_function, dataloader, epoch,
                        cuda_flag, optimizer=None, train=False):
    """
    Function to train/eval model on dataset for 1 epoch.

    Args:
        model: model
        loss_function: loss function
        dataloader: dataloader
        epoch: current epoch
        cuda_flag: cuda device
        optimizer: optimizer
        train: True(training)/False(validation), Default: False

    Returns:
        avg_loss, avg_accuracy, labels, preds, masks, avg_fscore,
        [alphas, alphas_f, alphas_b, vids]

    """

    device = torch.device('cuda')

    losses, preds, labels, masks = [], [], [], []
    alphas, alphas_f, alphas_b, vids = [], [], [], []
    max_sequence_len = []

    assert not train or optimizer != None
    if train:
        model.train()
    else:
        model.eval()

    seed_everything()
    for data in dataloader:
        if train:
            optimizer.zero_grad()

        textf, qmask, umask, label = [d.to(device) for d in data[:-1]] if cuda_flag else data[:-1]
        max_sequence_len.append(textf.size(0))

        log_prob, alpha, alpha_f, alpha_b, _ = model(textf, qmask, umask)
        # seq_len, batch, n_classes -> batch*seq_len, n_classes
        lp_ = log_prob.transpose(0,1).contiguous().view(-1, log_prob.size()[2])
        # batch*seq_len
        labels_ = label.view(-1)
        loss = loss_function(lp_, labels_, umask)

        # batch*seq_len
        pred_ = torch.argmax(lp_, 1)
        preds.append(pred_.data.cpu().numpy())
        labels.append(labels_.data.cpu().numpy())
        masks.append(umask.view(-1).cpu().numpy())

        losses.append(loss.item() * masks[-1].sum())
        if train:
            loss.backward()

            if args.tensorboard:
                for param in model.named_parameters():
                    writer.add_histogram(param[0], param[1].grad, epoch)

            optimizer.step()
        else:
            alphas += alpha
            alphas_f += alpha_f
            alphas_b += alpha_b
            vids +- data[-1]

    if preds != []:
        preds = np.concatenate(preds)
        labels = np.concatenate(labels)
        masks = np.concatenate(masks)
    else:
        return float('nan'), float('nan'), [], [], [], float('nan'), []

    avg_loss = round(np.sum(losses) / np.sum(masks), 4)
    avg_accuracy = round(accuracy_score(labels, preds, sample_weight=masks) * 100, 2)
    avg_fscore = round(f1_score(labels, preds, sample_weight=masks, average='weighted') * 100, 2)

    return avg_loss, avg_accuracy, labels, preds, masks, avg_fscore, \
           [alphas, alphas_f, alphas_b, vids]



if __name__ == '__main__':

    path = './checkpoints/DailyDialog/'

    parser = argparse.ArgumentParser()

    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='does not use GPU')
    parser.add_argument('--base-model', default='LSTM',
                        help='base recurrent model, must be one of LSTM/GRU')
    parser.add_argument('--graph-model', action='store_true', default=False,
                        help='whether to use graph model after recurrent encoding')
    parser.add_argument('--nodal-attention', action='store_true', default=False,
                        help='whether to use nodal attention in graph model: Equation-4,5,6 in Paper.')
    parser.add_argument('--windowp', type=int, default=7,
                        help='context window size for constructing edges in graph model for past utterances')
    parser.add_argument('--windowf', type=int, default=7,
                        help='context window size for constructing edges in graph model for future utterances')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate')
    parser.add_argument('--l2', type=float, default=0.0001, metavar='L2',
                        help='L2 regularization weight')
    parser.add_argument('--rec-dropout', type=float, default=0.1, metavar='rec_dropout',
                        help='rec_dropout rate')
    parser.add_argument('--dropout', type=float, default=0.2, metavar='dropout',
                        help='dropout rate')
    parser.add_argument('--batch-size', type=int, default=32, metavar='BS',
                        help='batch size')
    parser.add_argument('--epochs', type=int, default=70, metavar='E',
                        help='number of epochs')
    parser.add_argument('--class-weight', action='store_true', default=False,
                        help='use class weights')
    parser.add_argument('--active-listener', action='store_true', default=False,
                        help='active listener')
    parser.add_argument('--attention', default='general',
                        help='Attention type')
    parser.add_argument('--tensorboard', action='store_true', default=False,
                        help='Enables tensorboard log')

    args = parser.parse_args()
    print(args)

    args.cuda = torch.cuda.is_available() and not args.no_cuda
    if args.cuda:
        print('GPU available, Training on GPU...')
    else:
        print('GPU not available, Training on CPU...')

    if args.tensorboard:
        from tensorboardX import SummaryWriter

        writer = SummaryWriter()

    n_classes = 7
    cuda_flag = args.cuda
    n_epochs = args.epochs
    batch_size = args.batch_size
    device = torch.device('cuda' if cuda_flag else 'cpu')

    D_m, D_g = 100, 150
    D_p, D_e = 150, 100
    D_h, D_a = 100, 100

    graph_h = 100
    kernel_sizes = [3, 4, 5]
    glv_pretrained = np.load(open('dailydialog/glv_embedding_matrix', 'rb'), allow_pickle=True)
    vocab_size, embedding_dim = glv_pretrained.shape

    if args.graph_model:
        seed_everything()

        model = DialogueGCN_DailyModel(args.base_model,
                                       D_m, D_g, D_p, D_e, D_h, D_a, graph_h,
                                       n_speakers=2,
                                       max_seq_len=70,
                                       window_past=args.windowp,
                                       window_future=args.windowf,
                                       vocab_size=vocab_size,
                                       n_classes=n_classes,
                                       listener_state=args.active_listener,
                                       context_attention=args.attention,
                                       dropout=args.dropout,
                                       nodal_attention=args.nodal_attention,
                                       no_cuda=args.no_cuda
                                       )
        model.init_pretrained_embeddings(glv_pretrained)

        print(f'Graph NN with {args.base_model} as base model.')
        name = 'Graph'

    else:
        if args.base_model == 'DialogRNN':
            model = DialogRNNModel(D_m, D_g, D_p, D_e, D_h, D_a,
                                   n_classes=n_classes,
                                   listener_state=args.active_listener,
                                   context_attention=args.attention,
                                   dropout_rec=args.rec_dropout,
                                   dropout=args.dropout)

            print('Basic Dialog RNN Model.')

        elif args.base_model == 'GRU':
            model = GRUModel(D_m, D_e, D_h,
                             n_classes=n_classes,
                             dropout=args.dropout)

            print('Basic GRU Model.')

        elif args.base_model == 'LSTM':
            model = LSTMModel(D_m, D_e, D_h,
                              n_classes=n_classes,
                              dropout=args.dropout)

            print('Basic LSTM Model.')

        else:
            print('Base model must be one of LSTM/GRU/Transformer')
            raise NotImplementedError

        name = 'Base'

    if cuda_flag:
        model.to(device)

    loss_weights = torch.FloatTensor([1 / 0.0017,
                                      1 / 0.0034,
                                      1 / 0.1251,
                                      1 / 0.831,
                                      1 / 0.0099,
                                      1 / 0.0177,
                                      1 / 0.0112])

    if args.class_weight:
        if args.graph_model:
            loss_function = nn.NLLLoss(loss_weights.to(device) if cuda_flag else loss_weights).to(device)

        else:
            loss_function = MaskedNLLLoss(loss_weights.to(device) if cuda_flag else loss_weights).to(device)

    else:
        if args.graph_model:
            loss_function = nn.NLLLoss().to(device)
        else:
            loss_function = MaskedNLLLoss().to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)

    train_loader, valid_loader, test_loader = get_DailyDialogue_loader(
        'dailydialog/daily_dialogue.pkl',
        batch_size=batch_size,
        num_workers=0
    )

    all_fscore, all_precision, all_recall = [], [], []
    valid_acc, best_valid_acc = 0.0, 0.0
    save_name = ''

    for e in range(n_epochs):
        start_time = time.time()

        if args.graph_model:
            train_loss, train_acc, _, _, train_fscore, train_precision, train_recall = train_or_eval_graph_model(
                            model, loss_function, train_loader, e, cuda_flag, optimizer, train=True)

            valid_loss, valid_acc, _, _, valid_fscore, valid_precision, valid_recall = train_or_eval_graph_model(
                            model, loss_function, valid_loader, e, cuda_flag)

            all_fscore.append((train_fscore, valid_fscore))
            all_precision.append((train_precision, valid_precision))
            all_recall.append((train_recall, valid_recall))

##            torch.save({'model_state_dict': model.state_dict()},
##                       path + name + args.base_model + '_' + str(e) + '.pkl')

        else:
            train_loss, train_acc, _, _, _, train_fscore, _ = train_or_eval_model(
                model, loss_function, train_loader, e, cuda_flag, optimizer, True)

            valid_loss, valid_acc, _, _, _, valid_fscore, _ = train_or_eval_model(
                model, loss_function, valid_loader, e, cuda_flag)

            all_fscore.append((train_fscore, valid_fscore))

            save_name = name + args.base_model + '.pt'
            save_model(model, path, save_name)

##            torch.save({'model_state_dict': model.state_dict()},
##                       path + name + args.base_model + '_' + str(e), '.pkl')

        if args.tensorboard:
            writer.add_scalar('train: accuracy/loss', train_acc / train_loss, e)
            writer.add_scalar('valid: accuracy/loss', valid_acc / valid_loss, e)
            writer.add_scalar('test: accuracy/loss', test_acc / test_loss, e)

        print()
        print(f'Epoch: {e+1}, Time: {time_since(start_time)}')
        print('train_loss: {}, train_acc: {}, train_fscore: {}, train_precision: {}, train_recall: {}'.format(
            train_loss, train_acc, train_fscore, train_precision, train_recall))
        print('valid_loss: {}, valid_acc: {}, valid_fscore: {}, valid_precision: {}, valid_recall: {}'.format(
            valid_loss, valid_acc, valid_fscore, valid_precision, valid_recall))

        if valid_acc > best_valid_acc:
            save_name = name + args.base_model + '.pt'
            save_model(model, path, save_name)
            best_valid_acc = valid_acc
            print(f'Model saved at {path + save_name}...')

        print(f'{("=")*100}')


    # Training and validation performance
    index_max = all_fscore.index(max(all_fscore, key=lambda x: x[1]))
    print(f'F-Score(Tr/Va): {all_fscore[index_max]}')
    print(f'Precision(Tr/Va): {all_precision[index_max]}')
    print(f'Recall(Tr/Va): {all_recall[index_max]}')


    if args.tensorboard:
        writer.close()

    print()
    print('Loading best parameters for testing...')
    model.load_state_dict(torch.load(path + save_name))

    if args.graph_model:
        test_loss, test_acc, test_label, test_pred, test_fscore, test_precision, \
        test_recall = train_or_eval_graph_model(
            model, loss_function, test_loader, e, cuda_flag)
    else:
        test_loss, test_acc, test_label, test_pred, test_mask, \
                       test_fscore, attentions = train_or_eval_model(
                           model, loss_function, test_loader, e, cuda_flag)

##    print('test_loss: {}, test_acc: {}, test_fscore: {}, test_precision: {}, test_recall: {}'.format(
##            test_loss, test_acc, test_fscore, test_precision, test_recall))

    print()
    print('Test performance...')
    print(f'Loss: {test_loss}')
    print(f'Accuracy: {test_acc}')
    print(f'F-Score: {teset_fscore}')
    print(f'Precision: {test_precision}')
    print(f'Recall: {test_recall}')
