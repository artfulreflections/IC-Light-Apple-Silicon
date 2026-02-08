import numpy as np
import torch
import pytest
from unittest.mock import patch
import sys
import os

# Add parent dir to path so we can import utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestPytorch2Numpy:
    def test_quantized_output_uint8(self):
        """pytorch2numpy with quant=True returns uint8 arrays"""
        from utils import pytorch2numpy
        imgs = [torch.randn(3, 64, 64)]
        results = pytorch2numpy(imgs, quant=True)
        assert len(results) == 1
        assert results[0].dtype == np.uint8
        assert results[0].shape == (64, 64, 3)

    def test_float_output(self):
        """pytorch2numpy with quant=False returns float32 arrays"""
        from utils import pytorch2numpy
        imgs = [torch.randn(3, 64, 64)]
        results = pytorch2numpy(imgs, quant=False)
        assert results[0].dtype == np.float32
        assert results[0].min() >= 0.0
        assert results[0].max() <= 1.0


class TestNumpy2Pytorch:
    def test_shape_conversion(self):
        """numpy2pytorch converts HWC to NCHW"""
        from utils import numpy2pytorch
        imgs = [np.random.randint(0, 255, (64, 48, 3), dtype=np.uint8)]
        result = numpy2pytorch(imgs)
        assert result.shape == (1, 3, 64, 48)

    def test_normalization(self):
        """127 maps to approximately 0.0"""
        from utils import numpy2pytorch
        img = np.full((4, 4, 3), 127, dtype=np.uint8)
        result = numpy2pytorch([img])
        assert abs(result.mean().item()) < 0.01


class TestResizeAndCenterCrop:
    def test_output_dimensions(self):
        from utils import resize_and_center_crop
        img = np.random.randint(0, 255, (100, 80, 3), dtype=np.uint8)
        result = resize_and_center_crop(img, 64, 64)
        assert result.shape == (64, 64, 3)

    def test_upscale(self):
        from utils import resize_and_center_crop
        img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        result = resize_and_center_crop(img, 128, 128)
        assert result.shape == (128, 128, 3)


class TestResizeWithoutCrop:
    def test_output_dimensions(self):
        from utils import resize_without_crop
        img = np.random.randint(0, 255, (100, 80, 3), dtype=np.uint8)
        result = resize_without_crop(img, 64, 48)
        assert result.shape == (48, 64, 3)


class TestGetDevice:
    def test_returns_device(self):
        from utils import get_device
        device = get_device()
        assert isinstance(device, torch.device)
        assert device.type in ('cpu', 'cuda', 'mps')


class TestCreateSchedulers:
    def test_returns_three_schedulers(self):
        from utils import create_schedulers
        ddim, euler_a, dpmpp = create_schedulers()
        assert ddim is not None
        assert euler_a is not None
        assert dpmpp is not None


class TestClearGpuCache:
    def test_no_error_on_cpu(self):
        from utils import clear_gpu_cache
        device = torch.device('cpu')
        clear_gpu_cache(device)  # Should not raise


class TestParseCommonArgs:
    def test_defaults(self):
        from utils import parse_common_args
        with patch('sys.argv', ['test']):
            args = parse_common_args()
        assert args.host == '0.0.0.0'
        assert args.port == 7860
        assert args.model_dir == './models'
