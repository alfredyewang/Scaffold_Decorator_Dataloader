import os
import io
import json
import torch
import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset
import torchtext as tt  #
import collections
import re
from torch.utils.data import DataLoader



class ZINC(Dataset):

    def __init__(self, data_dir, create_data, **kwargs):

        super().__init__()
        self.data_dir = data_dir
        self.file_y='{}/y.smi'.format(data_dir)
        self.file_x1='{}/x1.smi'.format(data_dir)
        self.file_x2_1='{}/x2_1.smi'.format(data_dir)
        self.file_x2_2='{}/x2_2.smi'.format(data_dir)

        self.max_sequence_length_y = 74
        self.max_sequence_length_x1 = 134
        self.max_sequence_length_x2 = 60

        self.vocab_file = 'vocab.pth'
        self.data_file = 'data.json'


        if create_data:
            print("Creating data.")
            self._create_data()

        elif not os.path.exists(os.path.join(self.data_dir, self.data_file)):
            print("Data not found at %s. Generating..." % (os.path.join(self.data_dir, self.data_file)))
            self._create_data()
        else:
            self._load_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        idx = str(idx)

        return {
            'x1': np.asarray(self.data[idx]['x1']),
            'x2': np.asarray(self.data[idx]['x2']),
            'y': self.data[idx]['y']
        }

    @property
    def vocab_size(self):
        return len(self.vocabulary.get_itos())


    def _load_data(self, vocab=True):

        with open(os.path.join(self.data_dir, self.data_file), 'r') as file:
            self.data = json.load(file)
        if vocab:
            self.vocabulary=torch.load(os.path.join(self.data_dir, self.vocab_file))

    def _load_vocab(self):
        self.vocabulary=torch.load(os.path.join(self.data_dir, self.vocab_file))

    def _create_data(self):

        self._create_vocabulary()
        print(self.vocabulary.get_itos())
        data = defaultdict(dict)
        f_x1 = open(self.file_x1, "r")
        f_x2_1 = open(self.file_x2_1, "r")
        f_x2_2 = open(self.file_x2_2, "r")
        f_y = open(self.file_y, "r")

        for line_x1, line_x2_1, line_x2_2, line_y in zip(f_x1, f_x2_1, f_x2_2, f_y):
            line_x1 = line_x1.replace('[*]', "<pad>" * 30).strip()
            words_x1 = self._tokenize(line_x1, with_begin_and_end=True)
            words_x1 = words_x1 + ['<pad>'] * (self.max_sequence_length_x1 - len(words_x1))
            idx_x1 = self.vocabulary.lookup_indices(words_x1)

            line_x2_1 = line_x2_1.replace('*', "").strip()
            words_x2_1 = self._tokenize(line_x2_1, with_begin_and_end=False)
            words_x2_1 = words_x2_1 + ['<pad>'] * (int((self.max_sequence_length_x2/2)) - len(words_x2_1))
            idx_x2_1 = self.vocabulary.lookup_indices(words_x2_1)
            #
            line_x2_2 = line_x2_2.replace('*', "").strip()
            words_x2_2 = self._tokenize(line_x2_2, with_begin_and_end=False)
            words_x2_2 = words_x2_2 + ['<pad>'] * (int((self.max_sequence_length_x2/2)) - len(words_x2_2))
            idx_x2_2 = self.vocabulary.lookup_indices(words_x2_2)

            line_y = line_y.strip()
            words_y = self._tokenize(line_y, with_begin_and_end=True)
            words_y = words_y + ['<pad>'] * (self.max_sequence_length_y - len(words_y))
            idx_y = self.vocabulary.lookup_indices(words_y)

            idx_x2 = idx_x2_1 + idx_x2_2

            assert len(idx_x1) == self.max_sequence_length_x1
            assert len(idx_x2) == self.max_sequence_length_x2
            assert len(idx_y) == self.max_sequence_length_y

            id = len(data)
            data[id]['x1'] = idx_x1
            data[id]['x2'] = idx_x2
            data[id]['y'] = idx_y

        with io.open(os.path.join(self.data_dir, self.data_file), 'wb') as data_file:
            data = json.dumps(data, ensure_ascii=False)
            data_file.write(data.encode('utf8', 'replace'))

        self._load_data(vocab=False)

    def _tokenize(self, smiles, with_begin_and_end=True, start_token='$', end_token='^'):
        REGEXPS = {
            "brackets": re.compile(r"(\[[^\]]*\])"),
            "2_ring_nums": re.compile(r"(%\d{2})"),
            "brcl": re.compile(r"(Br|Cl)"),
            "pad": re.compile(r"(<pad>)")
        }
        REGEXP_ORDER = ["pad", "brackets", "2_ring_nums", "brcl"]
        """
        Tokenizes a SMILES string.
        :param smiles: A SMILES string.
        :param with_begin_and_end: Appends a begin token and prepends an end token.
        :return : A list with the tokenized version.
        """

        def split_by(smiles, regexps):
            if not regexps:
                return list(smiles)
            regexp = REGEXPS[regexps[0]]
            splitted = regexp.split(smiles)
            tokens = []
            for i, split in enumerate(splitted):
                if i % 2 == 0:
                    tokens += split_by(split, regexps[1:])
                else:
                    tokens.append(split)
            return tokens

        tokens = split_by(smiles, REGEXP_ORDER)
        if with_begin_and_end:
            tokens = ["{}".format(start_token)] + tokens + ["{}".format(end_token)]
        return tokens

    def _create_vocabulary(self):
        """
        Creates a vocabulary for the SMILES syntax.
        :param file: SMILES file PATH
        :param tokenizer: Tokenizer to use.
        :return: A vocabulary instance with all the tokens in the smiles_list.
        """
        counter_obj = collections.Counter()
        f = open(self.file_y, "r")
        for line in f:
            line = line.strip()
            words = self._tokenize(line, with_begin_and_end=False)
            counter_obj.update(words)
        f.close()
        vocabulary = tt.vocab.vocab(counter_obj, min_freq=1, specials=["<pad>", "$", "^"])
        torch.save(vocabulary,os.path.join(self.data_dir, self.vocab_file))
        self._load_vocab()


if __name__ == '__main__':
    dataset = ZINC(
        data_dir='./zinc/',
        create_data=False,
    )
    print(dataset.vocab_size)
    print(dataset.vocabulary.get_itos())
    print(dataset.vocabulary.lookup_indices(['C']))
    num_train= int(len(dataset) *0.8)
    num_test= len(dataset) -num_train

    train_data, test_data = torch.utils.data.random_split(dataset, [num_train, num_test])
    train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

