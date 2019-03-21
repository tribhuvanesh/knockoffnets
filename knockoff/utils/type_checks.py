import numpy as np
import torch


class TypeCheck(object):
    @staticmethod
    def single_image_blackbox_input(bb_input):
        if not isinstance(bb_input, np.ndarray):
            raise TypeError("Input must be an np.ndarray.")
        if bb_input.dtype != np.uint8:
            raise TypeError("Input must be an unit8 array.")
        if len(bb_input.shape) != 3:
            raise TypeError("Input must have three dims with (channel, height, width) elements.")

    @staticmethod
    def multiple_image_blackbox_input(bb_input):
        if not isinstance(bb_input, np.ndarray):
            raise TypeError("Input must be an np.ndarray.")
        if bb_input.dtype != np.uint8:
            raise TypeError("Input must be an unit8 array.")
        if len(bb_input.shape) != 4:
            raise TypeError("Input must have three dims with (num_samples, channel, height, width) elements.")

    @staticmethod
    def multiple_image_blackbox_input_tensor(bb_input):
        if not isinstance(bb_input, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor.")
        if bb_input.dtype != torch.float32:
            raise TypeError("Input must be a torch.float32 tensor.")
        if len(bb_input.shape) != 4:
            raise TypeError("Input must have three dims with (num_samples, channel, height, width) elements.")

    @staticmethod
    def single_label_int(label):
        if not isinstance(label, int):
            raise TypeError("Label must be an int.")

    @staticmethod
    def multiple_label_list_int(labels):
        if not isinstance(labels, list):
            raise TypeError("Labels must be a list.")
        for l in labels:
            if not isinstance(l, int):
                raise TypeError("Each label must be an int.")
