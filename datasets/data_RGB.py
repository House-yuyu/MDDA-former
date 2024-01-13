import os
from datasets.dataset_zx import Dataset_train, Dataset_val, Dataset_test, Fullsize_train, Fullsize_val
# from datasets.dataset_fusion import Dataset_train, Dataset_val


def get_training_data(rgb_dir, img_options):
    assert os.path.exists(rgb_dir)
    return Dataset_train(rgb_dir, img_options)


def get_validation_data(rgb_dir, img_options):
    assert os.path.exists(rgb_dir)
    return Dataset_val(rgb_dir, img_options)


def get_test_data(rgb_dir, img_options):
    assert os.path.exists(rgb_dir)
    return Dataset_test(rgb_dir, img_options)


def get_trainingfullsize_data(rgb_dir, img_options):
    assert os.path.exists(rgb_dir)
    return Fullsize_train(rgb_dir, img_options)


def get_validationfullsize_data(rgb_dir, img_options):
    assert os.path.exists(rgb_dir)
    return Fullsize_val(rgb_dir, img_options)


