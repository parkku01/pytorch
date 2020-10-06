import torch
import torchtext
import numpy as np
import config
from sequence_labeling.bilstm_with_att.bilstm_att import BiLSTMATT
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
            x, x_len = train_batch.input
            x = x.cuda()
            y = train_batch.target.cuda()
            model.zero_grad()
            pred = model.forward(x, x_len)

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
                    val_x, val_x_len = val_batch.input
                    val_x = val_x.cuda()
                    val_y = val_batch.target.cuda()
                    val_pred = model.forward(val_x, val_x_len)
                    val_loss.append(loss_function(val_pred, val_y).cpu().data.numpy())
                val_record.append({'step': step, 'loss': np.mean(val_loss)})
                print('epcoh {:02} - step {:06} - train_loss {:.4f} - val_loss {:.4f} '.format(
                    e, step, np.mean(losses), val_record[-1]['loss']))
                if e >= parameters['warmup_epoch']:
                    if val_record[-1]['loss'] <= max_loss:
                        save(m=model, info={'step': step, 'epoch': e, 'train_loss': np.mean(losses),
                                            'val_loss': val_record[-1]['loss']}, path=parameters['model_path'])
                        max_loss = val_record[-1]['loss']
                        no_improve_in_previous_epoch = False

def prediction(parameters):
    model, m_info = load(parameters['model_path'])
    model.eval()
    model.zero_grad()
    test_iter = parameters['test_iter']
    test_iter.init_epoch()
    total = 0
    correct = 0
    for test_batch in iter(test_iter):
        test_x, test_x_len = test_batch.input
        test_x = test_x.cuda()
        true = test_batch.target.data.numpy().tolist()
        pred = model.forward(test_x, test_x_len).transpose(1,2)
        pred = torch.argmax(pred, -1).cpu().data.numpy().tolist()
        for minibatch_p, minibatch_t in zip(pred, true):
            for p, t in zip(minibatch_p, minibatch_t):
                if t == parameters['data_helper'].target2idx['[CLS]']:
                    continue
                if p == t:
                    correct += 1
                total += 1

    print('accuracy:', correct/total, correct, total)

def prediction_demo(parameters):
    model, m_info = load(parameters['model_path'])
    model.eval()
    model.zero_grad()
    test_iter = parameters['test_iter']
    test_iter.init_epoch()
    dh = parameters['data_helper']

    for test_batch in iter(test_iter):
        test_x, test_x_len = test_batch.input
        test_x = test_x.cuda()
        pred = model.forward(test_x, test_x_len).transpose(1,2)
        pred = torch.argmax(pred, -1).cpu().data.numpy().tolist()
        true = test_batch.target.data.numpy().tolist()
        print(true)
        print(pred)
        test_x = test_x.cpu().data.numpy().tolist()
        print(test_x)
        result = dh.tokens2text(test_x, pred)
        print(result)

def save(m, info, path):
    torch.save(info, path+'best_model.info')
    torch.save(m, path+'best_model.m')


def load(path):
    m = torch.load(path+'best_model.m')
    info = torch.load(path+'best_model.info')
    return m, info


if __name__ == '__main__':
    dh = DataHelper()
    parameters = {
        'data_helper': dh,
        'embedding_dim': 100,
        'tokenizer': dh.tokenizer,
        'epoch': 1000,
        'model': None,
        'eval_every': 8,
        'loss_func': None,
        'optimizer': None,
        'train_iter': None,
        'val_iter': None,
        'test_iter': None,
        'early_stop': 20,
        'warmup_epoch': 2,
        'batch_size': 2,
        'learning_rate': 1e-3,
        'model_path': config.SEQ_LABEL_DIR+'/bilstm_with_att/models/news_sample/'
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
    parameters['test_iter'] = torchtext.data.BucketIterator(dataset=dh.test_data,
                                                           batch_size=1,
                                                           sort_key=lambda x: x.text.__len__(),
                                                           train=False,
                                                           sort=False)
    model = BiLSTMATT(embedding=None, parameters=parameters, lstm_layer=2, padding_idx=parameters['tokenizer'].vocab['[PAD]'], output_dim=len(dh.target2idx), hidden_dim=128).cuda()

    # loss_function = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([pos_w]).cuda())
    parameters['loss_func'] = torch.nn.CrossEntropyLoss(ignore_index=parameters['tokenizer'].vocab['[PAD]'])
    parameters['optimizer'] = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=parameters['learning_rate'])
    parameters['model'] = model
    training(parameters)
    # prediction(parameters)
    # prediction_demo(parameters)
