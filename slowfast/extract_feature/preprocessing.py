import torch as th
import random
import numpy as np
import math
from data_utils import convert_to_frac, convert_to_float


def get_start_end_idx(video_size, clip_size, clip_idx, num_clips):
    """
    Sample a clip of size clip_size from a video of size video_size and
    return the indices of the first and last frame of the clip. If clip_idx is
    -1, the clip is randomly sampled, otherwise uniformly split the video to
    num_clips clips, and select the start and end index of clip_idx-th video
    clip.
    Args:
        video_size (int): number of overall frames.
        clip_size (int): size of the clip to sample from the frames.
        clip_idx (int): if clip_idx is -1, perform random jitter sampling. If
            clip_idx is larger than -1, uniformly split the video to num_clips
            clips, and select the start and end index of the clip_idx-th video
            clip.
        num_clips (int): overall number of clips to uniformly sample from the
            given video for testing.
    Returns:
        start_idx (int): the start frame index.
        end_idx (int): the end frame index.
    """
    delta = max(video_size - clip_size, 0)
    if clip_idx == -1:
        # Random temporal sampling.
        start_idx = random.uniform(0, delta)
    else:
        # Uniformly sample the clip with the given index.
        start_idx = delta * clip_idx / num_clips
    end_idx = start_idx + clip_size - 1
    return start_idx, end_idx


def temporal_sampling(frames, start_idx, end_idx, num_samples):
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
    index = th.linspace(start_idx, end_idx, num_samples)
    index = th.clamp(index, 0, frames.shape[1] - 1).long()
    frames = th.index_select(frames, 1, index)
    return frames


class Normalize(object):

    def __init__(self, mean, std, device=th.device('cuda')):
        self.mean = th.FloatTensor(mean).view(1, 3, 1, 1, 1).float().to(device)
        self.std = th.FloatTensor(std).view(1, 3, 1, 1, 1).float().to(device)

    def __call__(self, tensor):
        tensor = (tensor - self.mean) / (self.std + 1e-8)
        return tensor


class Preprocessing(object):
    def __init__(self, type, cfg, target_fps=16, size=112,
                 clip_len='5', padding_mode='tile', min_num_clips=1, features_length=64):
        self.type = type # "3d"
        if type == '2d':
            self.norm = Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        elif type == '3d':
            # self.norm = Normalize(
            #   mean=[110.6, 103.2, 96.3], std=[1.0, 1.0, 1.0])
            self.norm = Normalize(
                mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])
        self.target_fps = target_fps # 30
        self.num_frames = cfg.DATA.NUM_FRAMES # 32
        self.sampling_rate = cfg.DATA.SAMPLING_RATE # 2
        self.cfg = cfg
        self.size = size # 224
        self.clip_len = clip_len # 2
        self.padding_mode = padding_mode # tile
        self.min_num_clips = min_num_clips # 1
        self.features_length = features_length # 64, 128, 256

    def _tile(self, a, dim, n_tile):
        init_dim = a.size(dim)
        repeat_idx = [1] * a.dim()
        repeat_idx[dim] = n_tile
        a = a.repeat(*(repeat_idx))
        order_index = th.LongTensor(
            np.concatenate(
                [init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
        return th.index_select(a, dim, order_index)

    def _pad_frames(self, tensor, mode='tile', value=0):
        # print(f"Target fps {self.target_fps} not satisfied, padding....")
        # self.features_length 배수에 맞게 pad
        n = self.features_length - len(tensor) % self.features_length
        if n == self.features_length:
            return tensor
        if mode == "constant":
            z = th.ones(n, tensor.shape[1], tensor.shape[2], tensor.shape[3],
                        dtype=th.uint8)
            z *= value
            return th.cat((tensor, z), 0)
        elif mode == "tile":
            z = th.cat(n * [tensor[-1:, :, :, :]])
            return th.cat((tensor, z), 0)
        else:
            raise NotImplementedError(
                f'Mode {mode} not implemented in _pad_frames.')

    def _pad_clips(self, tensor, mode='tile', value=0):
        # print(f"clip length {self.clip_len} not satisfied, padding....")
        clip_len = convert_to_frac(self.clip_len)
        assert len(clip_len) > 0
        if len(clip_len) == 1:
            clip_len = clip_len[0]
            n = clip_len - len(tensor) % clip_len
            if n == clip_len:
                return tensor
        else:
            clip_len_nom, clip_len_denom = clip_len[0], clip_len[1]
            frame_num = tensor.shape[1]
            assert frame_num*clip_len_nom % clip_len_denom == 0
            n = clip_len_nom - len(tensor) % clip_len_nom
            if n == clip_len_nom:
                return tensor

        if mode == "constant":
            z = th.ones(
                n, tensor.shape[1], tensor.shape[2],
                tensor.shape[3], tensor.shape[4],
                dtype=th.uint8)
            z *= value
            return th.cat((tensor, z), 0)
        elif mode == "tile":
            z = th.cat(n * [tensor[-1:, :, :, :, :]])
            return th.cat((tensor, z), 0)
        else:
            raise NotImplementedError(
                f'Mode {mode} not implemented in _pad_frames.')

    def __call__(self, tensor, info):
        if self.type == '2d':
            tensor = tensor / 255.0
            tensor = self.norm(tensor)
        elif self.type == '3d':
            # print('info:', info)
            # print('tensor.shape:', tensor.shape)
            # self.features_length 배수에 맞게 pad
            tensor = self._pad_frames(tensor, self.padding_mode)
            # print('after _pad_frames:', tensor.shape)
            # (features_length, (depends on T), height, width, channel)
            tensor = tensor.view(self.features_length, -1, self.size, self.size, 3)
            # print('after view:', tensor.shape)
            
            fps = info["fps"]  # .item()
            start_idx, end_idx = get_start_end_idx(
                tensor.shape[1],
                self.num_frames * self.sampling_rate * fps / self.target_fps,
                0,
                1,
            )
            # Perform temporal sampling from the decoded video.
            tensor = temporal_sampling(
                tensor, start_idx, end_idx, self.num_frames)
            # B T H W C
            # clips = clips.transpose(1, 2)
            # print('after temporal_sampling clips.shape:', tensor.shape)
        return tensor
