# Task Plan: IC-Light Improvement Roadmap

## Goal
Prioritize and implement UI/UX enhancements and feature improvements for IC-Light Apple Silicon fork, organized by t-shirt size estimates for planning.

## Current Phase
Phase 1 (Discovery & Prioritization)

## Completed Work (Prior Sessions)
- [x] Progress bar with `track_tqdm=True`
- [x] Info tooltips on all sliders/controls with value-range descriptions
- [x] Show/Hide help tips toggle (JS-based, targets sibling of `[data-testid='block-info']`)
- [x] Pipeline stages accordion (under output gallery)
- [x] Enhanced progress descriptions per stage
- [x] TECHNICAL.md documentation
- [x] CLAUDE.md audit and update
- [x] Auto-save outputs with timestamps
- [x] CLI arguments (--host, --port, --model-dir, --model, --output-dir)
- [x] Error handling with gr.Error for OOM and unexpected errors

---

## Improvement Backlog

### S (Small) - 1-2 hours each

#### S1: Randomize Seed Button
- Add a "Randomize" button next to the Seed field
- One click generates a random seed value
- Common UX pattern in SD interfaces that users expect
- **Files:** gradio_demo.py, gradio_demo_bg.py
- **Status:** pending

#### S2: Copy Seed from Output
- Display the seed used for each generated image in the gallery caption or as metadata
- Enables reproducibility when users find a result they like
- **Files:** gradio_demo.py, gradio_demo_bg.py
- **Status:** pending

#### S3: Aspect Ratio Presets
- Add a dropdown: "Portrait (512x640)", "Square (512x512)", "Landscape (640x512)", "Custom"
- Auto-sets width/height sliders, reducing manual adjustment
- **Files:** gradio_demo.py, gradio_demo_bg.py
- **Status:** pending

#### S4: Relight Button Disable During Processing
- Disable the Relight button while generation is in progress
- Prevents accidental double-clicks that queue duplicate jobs
- **Files:** gradio_demo.py, gradio_demo_bg.py
- **Status:** pending

#### S5: Dark/Light Theme Toggle
- Add a theme toggle or use `gr.Blocks(theme=...)` with Gradio's built-in themes
- Improves usability in different lighting environments
- **Files:** gradio_demo.py, gradio_demo_bg.py
- **Status:** pending

#### S6: Preset Configurations (Quick Settings)
- Add a dropdown with preset combos: "Fast Draft" (15 steps, 1.0 scale), "Balanced" (25 steps, 1.5 scale), "High Quality" (40 steps, 2.0 scale)
- Auto-fills steps, CFG, highres scale, and denoise values
- Reduces cognitive load for new users
- **Files:** gradio_demo.py, gradio_demo_bg.py
- **Status:** pending

### M (Medium) - 3-6 hours each

#### M1: Before/After Comparison View
- Add a slider or side-by-side toggle to compare input vs output
- Critical for relighting evaluation - users need to see what changed
- Could use Gradio's `gr.ImageSlider` or custom CSS overlay
- **Files:** gradio_demo.py, gradio_demo_bg.py
- **Status:** pending

#### M2: Generation History Panel
- Show thumbnails of recent generations with their settings (prompt, seed, steps)
- Click to reload settings from a previous generation
- Stored in-memory or as JSON alongside saved outputs
- **Files:** gradio_demo.py, gradio_demo_bg.py, utils.py
- **Status:** pending

#### M3: Prompt History/Favorites
- Autocomplete from previously used prompts
- Star/favorite button to save commonly used prompt+setting combos
- Store in a local JSON file
- **Files:** gradio_demo.py, gradio_demo_bg.py, utils.py
- **Status:** pending

#### M4: Image Download Button with Metadata
- Add explicit download button for output images
- Embed generation parameters (prompt, seed, steps, CFG) in PNG EXIF/metadata
- Enables sharing settings alongside results
- **Files:** gradio_demo.py, gradio_demo_bg.py, utils.py
- **Status:** pending

#### M5: Responsive Mobile Layout
- Current two-column layout breaks on mobile/tablet
- Add CSS media queries for single-column layout on narrow screens
- Move output gallery below inputs on mobile
- **Files:** gradio_demo.py, gradio_demo_bg.py
- **Status:** pending

#### M6: Tabbed Interface (Combine Both Demos)
- Merge gradio_demo.py and gradio_demo_bg.py into a single app with tabs
- "Text Conditioned" tab and "Background Conditioned" tab
- Share models in memory (currently loaded twice if running both)
- **Files:** new combined file or refactor existing
- **Status:** pending

### L (Large) - 1-2 days each

#### L1: LoRA Support
- Add LoRA loading UI (file upload or path input)
- Apply LoRA weights on top of base model + IC-Light offsets
- Enable custom art styles without changing base model
- **Files:** utils.py, gradio_demo.py, gradio_demo_bg.py
- **Status:** pending

#### L2: Batch Processing Mode
- Add a "Batch" tab that accepts a folder of images
- Process all images with the same settings
- Progress bar shows overall batch progress
- Output to a timestamped subfolder
- **Files:** utils.py, gradio_demo.py (or new batch script)
- **Status:** pending

#### L3: CPU Offloading for Memory Efficiency
- Move idle models to CPU when not in use
- Load back to GPU/MPS on demand
- Reduces VRAM usage from ~3.2GB to ~1GB during idle
- Critical for 8GB M1/M2 Macs
- **Files:** utils.py
- **Status:** pending

#### L4: REST API Endpoint
- Add a FastAPI/Flask endpoint alongside Gradio
- Accept JSON with image + settings, return relit image
- Enable integration with other tools (Photoshop plugins, automation)
- **Files:** new api.py, utils.py
- **Status:** pending

### XL (Extra Large) - 3+ days each

#### XL1: ControlNet Integration
- Add depth/canny/pose conditioning alongside IC-Light
- Better structure preservation during relighting
- Requires loading additional models
- **Files:** utils.py, gradio_demo.py, gradio_demo_bg.py
- **Status:** pending

#### XL2: Real-time Preview Mode
- Show low-res preview as user adjusts sliders (before clicking Relight)
- Use fewer steps (5-8) and smaller resolution for quick feedback
- Update on slider release, not during drag
- **Files:** gradio_demo.py, gradio_demo_bg.py
- **Status:** pending

---

## Prioritization Matrix (UI/UX Impact vs Effort)

| Priority | Items | Rationale |
|----------|-------|-----------|
| **Do First** | S1, S3, S6, M1 | High UX impact, low-medium effort, core user workflows |
| **Do Next** | S2, S4, M4, M6 | Quality-of-life, reduce friction, consolidate apps |
| **Do Later** | M2, M3, M5, L1, L3 | Nice-to-have, power user features, optimization |
| **Backlog** | S5, L2, L4, XL1, XL2 | Ambitious features, large scope, lower priority |

## Key Questions
1. Which items does the user want to tackle first?
2. Should the two demos be merged (M6) before adding features to both?
3. Is mobile support (M5) important for the target audience?
4. Is LoRA support (L1) a priority for the user's workflow?

## Decisions Made
| Decision | Rationale |
|----------|-----------|
| Use t-shirt sizing (S/M/L/XL) | Familiar estimation framework, quick to scan |
| Prioritize UX over features | Users need polish on existing functionality before new features |
| Recommend S1+S3+S6+M1 first | Biggest user experience wins for least effort |

## Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|
| (none yet) | - | - |

## Notes
- Update phase status as you progress: pending → in_progress → complete
- Re-read this plan before major decisions
- All estimates assume Apple Silicon/MPS as primary target
