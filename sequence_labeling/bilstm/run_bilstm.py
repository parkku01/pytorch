import torch
import torchtext
import numpy as np
from sequence_labeling.bilstm.bilstm import BiLSTM
from utils.news_data_helper import DataHelper

def training(parameters):
    step = 0
    max_loss = 1e5
    no_improve_epoch = 0
    no_improve_in_previous_epoch = False
    fine_tuning = False
    train_record = []
    val_record = []
    losses = []

    model = parameters['model']
    train_iter = parameters['train_iter']
    val_iter = parameters['val_iter']
    loss_function = parameters['loss_func']
    optimizer = parameters['optimizer']

    for e in range(parameters['epoch']):
        if e >= parameters['warmup_epoch']:
            if no_improve_in_previous_epoch:
                no_improve_epoch += 1
                if no_improve_epoch >= parameters['early_stop']:
                    break
            else:
                no_improve_epoch = 0
            no_improve_in_previous_epoch = True
        if not fine_tuning and e >= parameters['warmup_epoch']:
            model.embedding.weight.requires_grad = True
            fine_tuning = True

        train_iter.init_epoch()
        for train_batch in iter(train_iter):
            step += 1
            model.train()
            x = train_batch.input.cuda()
            y = train_batch.target.cuda()
            model.zero_grad()
            pred = model.forward(x)

            loss = loss_function(pred, y)
            losses.append(loss.cpu().data.numpy())
            train_record.append(loss.cpu().data.numpy())
            loss.backward()
            optimizer.step()
            if step % parameters['eval_every'] == 0:
                model.eval()
                model.zero_grad()
                val_loss = []
                for val_batch in iter(val_iter):
                    val_x = val_batch.input.cuda()
                    val_y = val_batch.target.cuda()
                    val_pred = model.forward(val_x)
                    val_loss.append(loss_function(val_pred, val_y).cpu().data.numpy())
                val_record.append({'step': step, 'loss': np.mean(val_loss)})
                print('epcoh {:02} - step {:06} - train_loss {:.4f} - val_loss {:.4f} '.format(
                    e, step, np.mean(losses), val_record[-1]['loss']))
                if e >= parameters['warmup_epoch']:
                    if val_record[-1]['loss'] <= max_loss:
                        save(m=model, info={'step': step, 'epoch': e, 'train_loss': np.mean(losses),
                                            'val_loss': val_record[-1]['loss']})
                        max_loss = val_record[-1]['loss']
                        no_improve_in_previous_epoch = False

def save(m, info):
    torch.save(info, 'best_model.info')
    torch.save(m, 'best_model.m')


def load():
    m = torch.load('best_model.m')
    info = torch.load('best_model.info')
    return m, info


if __name__ == '__main__':
    dh = DataHelper()
    parameters = {
        'embedding_dim': 100,
        'tokenizer': dh.tokenizer,
        'epoch': 10,
        'model': None,
        'eval_every': 2,
        'loss_func': None,
        'optimizer': None,
        'train_iter': None,
        'val_iter': None,
        'early_stop': 1,
        'warmup_epoch': 2,
        'batch_size': 2,
        'learning_rate': 1e-3
    }

    parameters['train_iter'] = torchtext.data.BucketIterator(dataset=dh.train_data,
                                               batch_size=parameters['batch_size'],
                                               sort_key=lambda x: x.text.__len__(),
                                               shuffle=True,
                                               sort=False)
    parameters['val_iter'] = torchtext.data.BucketIterator(dataset=dh.valid_data,
                                             batch_size=parameters['batch_size'],
                                             sort_key=lambda x: x.text.__len__(),
                                             train=False,
                                             sort=False)
    model = BiLSTM(embedding=None, parameters=parameters, lstm_layer=2, padding_idx=parameters['tokenizer'].vocab['[PAD]'], output_dim=len(dh.target2idx), hidden_dim=128).cuda()

    # loss_function = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([pos_w]).cuda())
    parameters['loss_func'] = torch.nn.CrossEntropyLoss(ignore_index=parameters['tokenizer'].vocab['[PAD]'])
    parameters['optimizer'] = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=parameters['learning_rate'])
    parameters['model'] = model
    training(parameters)
