import config
from torchtext import data
from transformers import BertTokenizer


class DataHelper(object):
    def __init__(self):
        self.tokenizer = BertTokenizer(config.RSC_DIR + '/char_vocab/vocab.txt', do_lower_case=False, do_basic_tokenize=False)
        self.data_load()

    def data_load(self):
        train_file = config.RSC_DIR+'/sbd/news/news_train.txt'
        valid_file = config.RSC_DIR + '/sbd/news/news_valid.txt'
        test_file = config.RSC_DIR + '/sbd/news/news_test.txt'
        target_vocab = config.RSC_DIR + '/sbd/news/target_vocab.txt'
        self.vocab_load(target_vocab)
        self.train_data = self.file_load(train_file)
        self.valid_data = self.file_load(valid_file)
        self.test_data = self.file_load(test_file)

    def vocab_load(self, file_path):
        self.target2idx = {}
        self.idx2target = {}
        with open(file_path, encoding='utf-8') as f:
            for token in f:
                token = token.strip()
                idx = len(self.target2idx)
                self.target2idx[token] = idx
                self.idx2target[idx] = token

    def text2tokens(self, text):
        tokens = []
        for sen in text.split(' / '):
            tokens += self.tokenizer.tokenize(sen)
        return tokens

    def text2targets(self, text):
        targets = []
        for sen in text.split(' / '):
            targets += self.text2target(self.tokenizer.tokenize(sen))
        return targets

    def tokens2text(self, minibatch_input, minibatch_pred):
        result = []
        for input, pred in zip(minibatch_input, minibatch_pred):
            ids = []
            batch_result = []
            for i, p in zip(input, pred):
                if i == self.tokenizer.vocab['[CLS]']:
                    continue
                elif p == self.target2idx['O']:
                    ids.append(i)
                    tokens = self.tokenizer.convert_ids_to_tokens(ids)
                    batch_result.append(self.tokenizer.convert_tokens_to_string(tokens))
                    ids = []
                else:
                    ids.append(i)
            result.append(batch_result)
        return result

    def text2target(self, tokens):
        target = []
        for i, token in enumerate(tokens):
            if i == 0:
                target.append('B')
            elif i == len(tokens)-1:
                target.append('O')
            else:
                target.append('I')
        return target

    def file_load(self, file_path):
        input = data.Field(sequential=True, use_vocab=True, tokenize=self.text2tokens, init_token='[CLS]', pad_token='[PAD]', unk_token = '[UNK]', lower=False, batch_first=True)
        target = data.Field(sequential=True, use_vocab=True, tokenize=self.text2targets, init_token='[CLS]', pad_token='[PAD]', unk_token = '[UNK]', lower=False, batch_first=True)

        dataset = data.TabularDataset(path=file_path, format='tsv',
                                    fields=[('input', input), ('target', target)], skip_header=False)

        input.build_vocab(dataset)
        target.build_vocab(dataset)
        input.vocab.stoi = self.tokenizer.vocab
        input.vocab.itos = self.tokenizer.ids_to_tokens
        target.vocab.stoi = self.target2idx
        target.vocab.itos = self.idx2target

        return dataset

if __name__ == '__main__':
    dh = DataHelper()
    tokens = dh.tokenizer.tokenize('"삼전(삼성전자) 팔고 갓슬라 (테슬라)로 갈아탈까요?"')
    print(tokens)
    ids = dh.tokenizer.convert_tokens_to_ids(tokens)
    print(type(ids), ids)
    convert_tokens = dh.tokenizer.convert_ids_to_tokens(ids)
    print(convert_tokens)
    print(dh.tokenizer.convert_tokens_to_string(convert_tokens))

