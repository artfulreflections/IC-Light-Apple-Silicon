import numpy as np
import torch
import pytest
from unittest.mock import patch, MagicMock
import sys
import os
import tempfile

# Add parent dir to path so we can import utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# --- pytorch2numpy ---

class TestPytorch2Numpy:
    def test_quantized_output_uint8(self):
        from utils import pytorch2numpy
        imgs = [torch.randn(3, 64, 64)]
        results = pytorch2numpy(imgs, quant=True)
        assert len(results) == 1
        assert results[0].dtype == np.uint8
        assert results[0].shape == (64, 64, 3)

    def test_float_output(self):
        from utils import pytorch2numpy
        imgs = [torch.randn(3, 64, 64)]
        results = pytorch2numpy(imgs, quant=False)
        assert results[0].dtype == np.float32
        assert results[0].min() >= 0.0
        assert results[0].max() <= 1.0

    def test_multiple_images(self):
        from utils import pytorch2numpy
        imgs = [torch.randn(3, 32, 32) for _ in range(4)]
        results = pytorch2numpy(imgs)
        assert len(results) == 4
        for r in results:
            assert r.shape == (32, 32, 3)
            assert r.dtype == np.uint8

    def test_quantized_clipping(self):
        from utils import pytorch2numpy
        # Large values should be clipped to 255
        imgs = [torch.ones(3, 4, 4) * 10.0]
        results = pytorch2numpy(imgs, quant=True)
        assert results[0].max() == 255

    def test_single_pixel(self):
        from utils import pytorch2numpy
        imgs = [torch.zeros(3, 1, 1)]
        results = pytorch2numpy(imgs)
        assert results[0].shape == (1, 1, 3)


# --- numpy2pytorch ---

class TestNumpy2Pytorch:
    def test_shape_conversion(self):
        from utils import numpy2pytorch
        imgs = [np.random.randint(0, 255, (64, 48, 3), dtype=np.uint8)]
        result = numpy2pytorch(imgs)
        assert result.shape == (1, 3, 64, 48)

    def test_normalization(self):
        from utils import numpy2pytorch
        img = np.full((4, 4, 3), 127, dtype=np.uint8)
        result = numpy2pytorch([img])
        assert abs(result.mean().item()) < 0.01

    def test_multiple_images(self):
        from utils import numpy2pytorch
        imgs = [np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8) for _ in range(3)]
        result = numpy2pytorch(imgs)
        assert result.shape == (3, 3, 32, 32)

    def test_zero_maps_negative(self):
        from utils import numpy2pytorch
        img = np.zeros((4, 4, 3), dtype=np.uint8)
        result = numpy2pytorch([img])
        assert result.min().item() < 0

    def test_255_maps_positive(self):
        from utils import numpy2pytorch
        img = np.full((4, 4, 3), 255, dtype=np.uint8)
        result = numpy2pytorch([img])
        assert result.max().item() > 0


# --- Roundtrip conversion ---

class TestConversionRoundtrip:
    def test_roundtrip_preserves_shape(self):
        from utils import pytorch2numpy, numpy2pytorch
        original = [np.random.randint(0, 255, (64, 48, 3), dtype=np.uint8)]
        tensor = numpy2pytorch(original)
        back = pytorch2numpy(tensor)
        assert back[0].shape == original[0].shape
        assert back[0].dtype == original[0].dtype

    def test_roundtrip_approximate_values(self):
        from utils import pytorch2numpy, numpy2pytorch
        original = [np.full((4, 4, 3), 127, dtype=np.uint8)]
        tensor = numpy2pytorch(original)
        back = pytorch2numpy(tensor)
        # Should be close to original (within rounding)
        assert np.abs(back[0].astype(int) - original[0].astype(int)).max() <= 2


# --- resize_and_center_crop ---

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

    def test_landscape_to_square(self):
        from utils import resize_and_center_crop
        img = np.random.randint(0, 255, (50, 200, 3), dtype=np.uint8)
        result = resize_and_center_crop(img, 64, 64)
        assert result.shape == (64, 64, 3)

    def test_portrait_to_landscape(self):
        from utils import resize_and_center_crop
        img = np.random.randint(0, 255, (200, 50, 3), dtype=np.uint8)
        result = resize_and_center_crop(img, 128, 64)
        assert result.shape == (64, 128, 3)

    def test_small_image(self):
        from utils import resize_and_center_crop
        img = np.random.randint(0, 255, (4, 4, 3), dtype=np.uint8)
        result = resize_and_center_crop(img, 64, 64)
        assert result.shape == (64, 64, 3)

    def test_preserves_dtype(self):
        from utils import resize_and_center_crop
        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        result = resize_and_center_crop(img, 32, 32)
        assert result.dtype == np.uint8


# --- resize_without_crop ---

class TestResizeWithoutCrop:
    def test_output_dimensions(self):
        from utils import resize_without_crop
        img = np.random.randint(0, 255, (100, 80, 3), dtype=np.uint8)
        result = resize_without_crop(img, 64, 48)
        assert result.shape == (48, 64, 3)

    def test_upscale(self):
        from utils import resize_without_crop
        img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        result = resize_without_crop(img, 256, 256)
        assert result.shape == (256, 256, 3)

    def test_non_square(self):
        from utils import resize_without_crop
        img = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
        result = resize_without_crop(img, 50, 25)
        assert result.shape == (25, 50, 3)

    def test_preserves_dtype(self):
        from utils import resize_without_crop
        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        result = resize_without_crop(img, 32, 32)
        assert result.dtype == np.uint8


# --- get_device ---

class TestGetDevice:
    def test_returns_device(self):
        from utils import get_device
        device = get_device()
        assert isinstance(device, torch.device)
        assert device.type in ('cpu', 'cuda', 'mps')


# --- create_schedulers ---

class TestCreateSchedulers:
    def test_returns_three_schedulers(self):
        from utils import create_schedulers
        ddim, euler_a, dpmpp = create_schedulers()
        assert ddim is not None
        assert euler_a is not None
        assert dpmpp is not None

    def test_scheduler_types(self):
        from utils import create_schedulers
        from diffusers import DDIMScheduler, EulerAncestralDiscreteScheduler, DPMSolverMultistepScheduler
        ddim, euler_a, dpmpp = create_schedulers()
        assert isinstance(ddim, DDIMScheduler)
        assert isinstance(euler_a, EulerAncestralDiscreteScheduler)
        assert isinstance(dpmpp, DPMSolverMultistepScheduler)

    def test_scheduler_train_timesteps(self):
        from utils import create_schedulers
        ddim, euler_a, dpmpp = create_schedulers()
        assert ddim.config.num_train_timesteps == 1000
        assert euler_a.config.num_train_timesteps == 1000
        assert dpmpp.config.num_train_timesteps == 1000


# --- get_default_scheduler ---

class TestGetDefaultScheduler:
    def test_mps_returns_ddim(self):
        from utils import get_default_scheduler, create_schedulers
        ddim, _, dpmpp = create_schedulers()
        mps_device = torch.device('mps')
        result = get_default_scheduler(mps_device, ddim, dpmpp)
        assert result is ddim

    def test_cuda_returns_dpmpp(self):
        from utils import get_default_scheduler, create_schedulers
        ddim, _, dpmpp = create_schedulers()
        cuda_device = torch.device('cuda')
        result = get_default_scheduler(cuda_device, ddim, dpmpp)
        assert result is dpmpp

    def test_cpu_returns_dpmpp(self):
        from utils import get_default_scheduler, create_schedulers
        ddim, _, dpmpp = create_schedulers()
        cpu_device = torch.device('cpu')
        result = get_default_scheduler(cpu_device, ddim, dpmpp)
        assert result is dpmpp


# --- clear_gpu_cache ---

class TestClearGpuCache:
    def test_no_error_on_cpu(self):
        from utils import clear_gpu_cache
        device = torch.device('cpu')
        clear_gpu_cache(device)  # Should not raise


# --- save_outputs ---

class TestSaveOutputs:
    def test_creates_output_directory(self):
        from utils import save_outputs
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, 'new_subdir')
            img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            save_outputs([img], output_dir, prefix='test')
            assert os.path.isdir(output_dir)

    def test_saves_correct_number_of_files(self):
        from utils import save_outputs
        with tempfile.TemporaryDirectory() as tmpdir:
            imgs = [np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8) for _ in range(3)]
            paths = save_outputs(imgs, tmpdir, prefix='test')
            assert len(paths) == 3
            for p in paths:
                assert os.path.exists(p)

    def test_filename_format(self):
        from utils import save_outputs
        with tempfile.TemporaryDirectory() as tmpdir:
            img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
            paths = save_outputs([img], tmpdir, prefix='fc_relight')
            filename = os.path.basename(paths[0])
            assert filename.startswith('fc_relight_')
            assert filename.endswith('_00.png')

    def test_saved_image_is_valid_png(self):
        from utils import save_outputs
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmpdir:
            img = np.random.randint(0, 255, (64, 48, 3), dtype=np.uint8)
            paths = save_outputs([img], tmpdir, prefix='test')
            loaded = Image.open(paths[0])
            assert loaded.size == (48, 64)  # PIL size is (width, height)

    def test_returns_paths(self):
        from utils import save_outputs
        with tempfile.TemporaryDirectory() as tmpdir:
            img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
            paths = save_outputs([img], tmpdir, prefix='test')
            assert isinstance(paths, list)
            assert all(isinstance(p, str) for p in paths)


# --- parse_common_args ---

class TestParseCommonArgs:
    def test_defaults(self):
        from utils import parse_common_args
        with patch('sys.argv', ['test']):
            args = parse_common_args()
        assert args.host == '0.0.0.0'
        assert args.port == 7860
        assert args.model_dir == './models'
        assert args.model == 'stablediffusionapi/realistic-vision-v51'
        assert args.output_dir == './outputs'

    def test_custom_host_port(self):
        from utils import parse_common_args
        with patch('sys.argv', ['test', '--host', '127.0.0.1', '--port', '8080']):
            args = parse_common_args()
        assert args.host == '127.0.0.1'
        assert args.port == 8080

    def test_custom_model(self):
        from utils import parse_common_args
        with patch('sys.argv', ['test', '--model', 'runwayml/stable-diffusion-v1-5']):
            args = parse_common_args()
        assert args.model == 'runwayml/stable-diffusion-v1-5'

    def test_custom_output_dir(self):
        from utils import parse_common_args
        with patch('sys.argv', ['test', '--output-dir', '/tmp/my_outputs']):
            args = parse_common_args()
        assert args.output_dir == '/tmp/my_outputs'

    def test_custom_model_dir(self):
        from utils import parse_common_args
        with patch('sys.argv', ['test', '--model-dir', '/data/models']):
            args = parse_common_args()
        assert args.model_dir == '/data/models'

    def test_custom_description(self):
        from utils import parse_common_args
        with patch('sys.argv', ['test']):
            args = parse_common_args(description='Custom Description')
        # Should not raise; description is for help text only
        assert args is not None
