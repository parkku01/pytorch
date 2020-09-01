import config
from torchtext import data
from transformers import BertTokenizer


class DataHelper(object):
    def __init__(self):
        self.tokenizer = BertTokenizer(config.RSC_DIR + '/char_vocab/vocab.txt', do_lower_case=False)
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
        print()

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