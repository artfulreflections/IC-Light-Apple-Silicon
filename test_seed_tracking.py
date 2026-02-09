#!/usr/bin/env python3
"""Test script for seed tracking feature in batch generation."""

import numpy as np
import tempfile
import os
from utils import save_outputs

def test_seed_tracking():
    """Test that save_outputs correctly uses individual seeds for each image."""
    print("Testing seed tracking feature...")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create 3 test images
        images = [np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8) for _ in range(3)]
        seeds = [12345, 12346, 12347]

        # Test with seeds list (new feature)
        print("\n1. Testing with seeds list:")
        paths = save_outputs(images, tmpdir, prefix='test', seeds=seeds)

        print("   Generated files:")
        for path in paths:
            filename = os.path.basename(path)
            print(f'     {filename}')
            assert 'seed' in filename, f'Seed not in filename: {filename}'

        # Check that each file has the correct seed
        for i, path in enumerate(paths):
            filename = os.path.basename(path)
            expected_seed = str(seeds[i])
            assert f'seed{expected_seed}' in filename, f'Expected seed{expected_seed} in {filename}'

        print("   ✓ Seeds correctly included in filenames")

        # Test with single seed (backward compatibility)
        print("\n2. Testing with single seed (backward compatibility):")
        paths = save_outputs(images, tmpdir, prefix='test', seed=99999)

        print("   Generated files:")
        for path in paths:
            filename = os.path.basename(path)
            print(f'     {filename}')
            assert 'seed99999' in filename, f'Seed not in filename: {filename}'

        print("   ✓ Single seed backward compatibility works")

        # Test with no seed
        print("\n3. Testing with no seed:")
        paths = save_outputs(images, tmpdir, prefix='test')

        print("   Generated files:")
        for path in paths:
            filename = os.path.basename(path)
            print(f'     {filename}')
            # Should not have 'seed' in filename when no seed provided
            assert filename.count('seed') == 0, f'Unexpected seed in filename: {filename}'

        print("   ✓ No seed case works correctly")

    print("\n✅ All seed tracking tests passed!")

if __name__ == '__main__':
    test_seed_tracking()
