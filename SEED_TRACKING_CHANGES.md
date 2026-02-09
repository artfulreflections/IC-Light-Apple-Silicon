# Seed Tracking Implementation

## Overview
Implemented individual seed tracking for batch generation. When `num_samples > 1`, each generated image now has its own seed (seed, seed+1, seed+2, etc.) embedded in the filename.

## Changes Made

### 1. utils.py
Modified `save_outputs()` function to accept a `seeds` parameter:
- Added `seeds: Optional[list[int]]` parameter for per-image seed tracking
- Kept `seed: Optional[int]` for backward compatibility
- Updated filename format: `{prefix}_{timestamp}_seed{individual_seed}_{index:02d}.png`
- Each image gets its specific seed in the filename when `seeds` list is provided

### 2. gradio_demo.py (Text-conditioned relighting)
Modified `process()` function:
- Added seed tracking: `used_seeds = [int(seed) + i for i in range(num_samples)]`
- Updated return statement to include `used_seeds`: `return pytorch2numpy(pixels), used_seeds`

Modified `process_relight()` function:
- Updated to unpack `used_seeds` from `process()`: `results, used_seeds = process(...)`
- Updated `save_outputs()` call to use `seeds` parameter: `save_outputs(results, args.output_dir, prefix='fc_relight', seeds=used_seeds)`

### 3. gradio_demo_bg.py (Background-conditioned relighting)
Modified `process()` function:
- Added seed tracking: `used_seeds = [int(seed) + i for i in range(num_samples)]`
- Updated return statement to include `used_seeds`: `return pixels, [fg, bg], used_seeds`

Modified `process_relight()` function:
- Updated to unpack `used_seeds` from `process()`: `results, extra_images, used_seeds = process(...)`
- Updated `save_outputs()` call to use `seeds` parameter: `save_outputs(results, args.output_dir, prefix='fbc_relight', seeds=used_seeds)`

Modified `process_normal()` function:
- Updated to handle new 3-value return from `process()`: `pixels, _, _ = process(...)`
- Kept single `seed` parameter for `save_outputs()` since normal generation doesn't use batch mode

## Filename Examples

### Before
- `fc_relight_20250207_143022_00.png`
- `fc_relight_20250207_143022_01.png`
- `fc_relight_20250207_143022_02.png`

### After (with seed=12345, num_samples=3)
- `fc_relight_20250207_143022_seed12345_00.png`
- `fc_relight_20250207_143022_seed12346_01.png`
- `fc_relight_20250207_143022_seed12347_02.png`

## How Seeds Work in Batch Generation

When generating multiple images (`num_samples > 1`), the PyTorch generator is initialized with the base seed:
```python
rng = torch.Generator(device=device).manual_seed(int(seed))
```

The diffusers pipeline then uses this generator to produce `num_samples` images. Each image effectively uses a sequential seed:
- Image 0: seed
- Image 1: seed + 1
- Image 2: seed + 2
- etc.

This implementation tracks these sequential seeds and includes them in the saved filenames.

## Testing

Run the test script to verify functionality:
```bash
python test_seed_tracking.py
```

The test validates:
1. Seeds list parameter correctly embeds individual seeds in filenames
2. Backward compatibility with single seed parameter
3. No-seed case still works correctly

## Backward Compatibility

The changes maintain full backward compatibility:
- Old code using `seed=` parameter continues to work
- New code can use `seeds=` parameter for per-image tracking
- If both are provided, `seeds` takes precedence per image
