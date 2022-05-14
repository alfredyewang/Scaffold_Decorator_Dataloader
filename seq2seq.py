

import dataset as md
import torch.utils.data as tud
import os.path
import glob
import itertools as it

import vocabulary as mv

def load_sets(set_path):
    file_paths = [set_path]
    if os.path.isdir(set_path):
        file_paths = sorted(glob.glob("{}/*.smi".format(set_path)))

    for path in it.cycle(file_paths):  # stores the path instead of the set
        return list(read_csv_file(path, num_fields=2))

def read_csv_file(file_path, ignore_invalid=True, num=-1, num_fields=0):

    with open_file(file_path, "rt") as csv_file:
        for i, row in enumerate(csv_file):
            if i == num:
                break
            fields = row.rstrip().split("\t")
            if fields:
                if num_fields > 0:
                    fields = fields[0:num_fields]
                yield fields
            elif not ignore_invalid:
                yield None


def open_file(path, mode="r", with_gzip=False):

    open_func = open
    if path.endswith(".gz") or with_gzip:
        open_func = gzip.open
    return open_func(path, mode)


if __name__ == '__main__':
    scaffold_list, decoration_list = zip(*read_csv_file('zinc/zinc.smi', num_fields=2))
    vocabulary = mv.DecoratorVocabulary.from_lists(scaffold_list, decoration_list)
    training_sets = load_sets('zinc/zinc.smi')
    dataset = md.DecoratorDataset(training_sets, vocabulary=vocabulary)
    dataloader = tud.DataLoader(dataset, batch_size=128, shuffle=True, collate_fn=md.DecoratorDataset.collate_fn, drop_last=True)

    for epoch in range(2):
        for scaffold_batch, decorator_batch in dataloader:
            print(scaffold_batch[0])            # x index
            print(scaffold_batch[1])            # x length in this batch
            print(decorator_batch[0])           # y index
            print(decorator_batch[1])           # y length in this batch
            exit()


