# Seed Management Feature

## Overview

Added the ability to save and load favorite seeds across both demos (`gradio_demo.py` and `gradio_demo_bg.py`). Seeds are now embedded in output filenames and can be saved to a persistent JSON file for reuse.

## Features Implemented

### 1. Seed in Output Filenames

Generated images now include the seed in their filenames for easy reference:

**Format**: `{prefix}_{timestamp}_seed{seed}_{index}.png`

**Examples**:
- `fc_relight_20250208_143022_seed12345_00.png`
- `fbc_relight_20250208_143022_seed67890_00.png`
- `fbc_normal_20250208_143022_seed12345_00.png`

This makes it easy to find which seed produced an image you liked by looking at the filename.

### 2. Persistent Favorite Seeds

Seeds are saved to `favorite_seeds.json` in the output directory (default: `./outputs/`). The file persists across app restarts.

**JSON Format**:
```json
[
  {
    "seed": 12345,
    "label": "Golden sunset look",
    "timestamp": "2025-02-08T14:30:22.123456",
    "settings": {
      "steps": 25,
      "cfg": 2.0
    }
  },
  {
    "seed": 67890,
    "label": "Dramatic rim lighting",
    "timestamp": "2025-02-08T14:35:10.456789"
  }
]
```

### 3. UI Components

Both demos now have:

1. **"Load Favorite Seed" dropdown** - Select a previously saved seed to populate the Seed field
2. **"Seed Label" textbox** - Optional label for saving seeds (e.g., "Golden sunset")
3. **"💾 Save Seed" button** - Save the current seed to favorites

**UI Layout**:
```
┌─────────────────────────────────────────────┐
│ Seed: [12345]  [Randomize]                  │
├─────────────────────────────────────────────┤
│ Load Favorite Seed: [dropdown▼]             │
│ Seed Label: [____________]  [💾 Save Seed]  │
└─────────────────────────────────────────────┘
```

## Usage Examples

### Saving a Favorite Seed

1. Generate an image with seed `12345` that you really like
2. (Optional) Enter a label like "Golden sunset look" in the Seed Label field
3. Click "💾 Save Seed"
4. You'll see "✅ Saved seed 12345" confirmation

The seed is now saved to `./outputs/favorite_seeds.json` and appears in the dropdown.

### Loading a Favorite Seed

1. Open the "Load Favorite Seed" dropdown
2. Select a previously saved seed (e.g., "Golden sunset look (seed: 12345)")
3. The seed field automatically populates with `12345`
4. Generate a new image with those settings

### Finding Seeds from Past Images

If you find an old image you like:

1. Look at the filename: `fc_relight_20250208_143022_seed12345_00.png`
2. The seed is `12345`
3. Enter it in the Seed field to reproduce the result

## Implementation Details

### New Functions in `utils.py`

```python
# Load all saved favorite seeds
load_favorite_seeds(output_dir: str) -> list[dict[str, Any]]

# Save a seed to favorites (with optional label and settings)
save_favorite_seed(output_dir: str, seed: int, label: str = "", settings: Optional[dict] = None) -> bool

# Get formatted choices for Gradio dropdown
get_favorite_seeds_choices(output_dir: str) -> list[tuple[str, int]]

# Save outputs with seed in filename
save_outputs(images: list[np.ndarray], output_dir: str, prefix: str = 'relight', seed: Optional[int] = None) -> list[str]
```

### Event Handlers

Both demos implement:

```python
def handle_save_seed(current_seed, label):
    """Save the current seed to favorites with optional label"""
    # Saves to favorite_seeds.json
    # Refreshes dropdown choices
    # Returns success message

def handle_load_favorite(choice):
    """Load a seed from the dropdown selection"""
    # Parses "Label (seed: 12345)" format
    # Returns seed value for the seed field
```

## File Locations

- **Favorites file**: `./outputs/favorite_seeds.json` (or `{--output-dir}/favorite_seeds.json`)
- **Output images**: `./outputs/*.png` (with seed in filename)
- **Implementation**: `utils.py`, `gradio_demo.py`, `gradio_demo_bg.py`

## Testing

A test script is available at `test_seed_management.py` to verify:

- Saving seeds
- Loading seeds
- Updating existing seeds
- Dropdown choices generation
- Empty directory handling

Run (in virtual environment):
```bash
python test_seed_management.py
```

## Notes

- Seeds are saved immediately when you click "💾 Save Seed"
- Saving the same seed again updates the existing entry (doesn't duplicate)
- The dropdown refreshes automatically after saving
- Labels are optional - seeds default to "Seed 12345" if no label provided
- Settings are optional - can be added for future reference but not used yet
- The favorites file is human-readable JSON for easy manual editing

## Future Enhancements

Potential improvements for later:

1. **Load settings with seed** - Populate all UI fields (steps, CFG, etc.) when loading a favorite
2. **Delete favorites** - UI button to remove unwanted seeds from the list
3. **Import/export** - Share favorite seeds with others via JSON file
4. **Thumbnails** - Store thumbnail images with favorites for visual reference
5. **Search/filter** - Search favorites by label or sort by timestamp
6. **Categories** - Organize favorites into groups (portraits, landscapes, etc.)
