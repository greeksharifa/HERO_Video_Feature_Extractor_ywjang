import torch as th


class Normalize(object):

    def __init__(self, mean, std):
        self.mean = th.FloatTensor(mean).view(1, 3, 1, 1)
        self.std = th.FloatTensor(std).view(1, 3, 1, 1)

    def __call__(self, tensor):
        tensor = (tensor - self.mean) / (self.std + 1e-8)
        return tensor


class Preprocessing(object):

    def __init__(self, features_length):
        self.features_length = features_length # 64, 128, 256
        self.norm = Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711])

    def __call__(self, tensor):
        tensor = temporal_sampling(tensor, self.features_length)
        tensor = tensor / 255.0
        tensor = self.norm(tensor)
        return tensor


def temporal_sampling(frames, num_samples):
    """
    Given the start and end frame index, sample num_samples frames between
    the start and end with equal interval.
    Args:
        frames (tensor): a tensor of video frames, dimension is
            `batch_size` x `num video frames` x `height` x `width` x `channel`.
        start_idx (int): the index of the start frame.
        end_idx (int): the index of the end frame.
        num_samples (int): number of frames to sample. # 32
    Returns:
        frames (tersor): a tensor of temporal sampled video frames,
            dimension is
            `batch_size` x `num clip frames`` x `height` x `width` x `channel.
    """
    # print('frames.shape before:', frames.shape)
    num_frames = frames.shape[0]
    index = th.linspace(0, num_frames-1, num_samples)
    index = th.clamp(index, 0, num_frames - 1).long()
    frames = th.index_select(frames, 0, index)
    # print('frames.shape after :', frames.shape)
    return frames