# Seed Management UI Preview

## Before (Original)

```
┌─────────────────────────────────────────────────────┐
│ Images: [====1====]  Seed: [12345]  [Randomize]     │
│                                                      │
│ Image Width: [====512====]                          │
│ Image Height: [====640====]                         │
└─────────────────────────────────────────────────────┘
```

## After (With Seed Management)

```
┌──────────────────────────────────────────────────────────┐
│ Images: [====1====]  Seed: [12345]  [Randomize]          │
│                                                           │
│ Load Favorite Seed: [Golden sunset (seed: 12345) ▼]      │
│ Seed Label: [                    ]  [💾 Save Seed]       │
│                                                           │
│ Image Width: [====512====]                               │
│ Image Height: [====640====]                              │
└──────────────────────────────────────────────────────────┘
```

## Workflow Examples

### Example 1: Save a Good Result

```
User Action:
1. Generate image with seed 12345 → looks great!
2. Type "Golden sunset" in Seed Label
3. Click "💾 Save Seed"

Result:
✅ Saved seed 12345
Dropdown now shows: "Golden sunset (seed: 12345)"
```

### Example 2: Reuse a Favorite

```
User Action:
1. Open "Load Favorite Seed" dropdown
2. Select "Golden sunset (seed: 12345)"
3. Click Generate

Result:
Seed field auto-fills to 12345
Same lighting as before!
```

### Example 3: Find Old Image

```
User Action:
1. Find old image: fc_relight_20250208_143022_seed67890_00.png
2. See "seed67890" in filename
3. Enter 67890 in Seed field
4. Click Generate

Result:
Reproduces the exact same result
```

## Output Filename Format

### Old Format
```
fc_relight_20250208_143022_00.png
fbc_relight_20250208_143022_00.png
```

### New Format
```
fc_relight_20250208_143022_seed12345_00.png
fbc_relight_20250208_143022_seed67890_00.png
fbc_normal_20250208_143022_seed12345_00.png
```

**Benefits**:
- Instantly see which seed produced an image
- No need to remember or write down seeds
- Easy to reproduce favorite results

## Favorite Seeds JSON

Located at: `./outputs/favorite_seeds.json`

```json
[
  {
    "seed": 12345,
    "label": "Golden sunset look",
    "timestamp": "2025-02-08T14:30:22.123456"
  },
  {
    "seed": 67890,
    "label": "Dramatic rim lighting",
    "timestamp": "2025-02-08T14:35:10.456789"
  },
  {
    "seed": 99999,
    "label": "Soft natural light",
    "timestamp": "2025-02-08T14:40:05.789012"
  }
]
```

## UI Feedback Messages

```
When saving:
✅ Saved seed 12345          (Success)
⚠️ No seed to save           (No seed entered)
❌ Failed to save seed       (Write error)
```

## Integration with Existing Features

Works seamlessly with:
- ✅ Randomize button (still works as before)
- ✅ Example gallery (loads seed from examples)
- ✅ Quality presets (seeds saved with any quality settings)
- ✅ Both demos (text-conditioned and background-conditioned)
- ✅ All schedulers (DDIM, Euler a, DPM++)
- ✅ Auto-save to outputs/ directory

## Technical Notes

- **Persistence**: Favorites survive app restarts
- **Updates**: Saving same seed updates existing entry
- **Performance**: No performance impact (saves async)
- **Storage**: Minimal disk usage (~1KB per 50 favorites)
- **Compatibility**: Works with all existing CLI args (`--output-dir`, etc.)
