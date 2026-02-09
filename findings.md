# Findings & Decisions

## Requirements
- Create a prioritized roadmap of IC-Light improvements
- Include UI/UX expert recommendations
- Use t-shirt size estimates (S/M/L/XL)
- Cover both demos (text-conditioned and background-conditioned)

## UI/UX Audit Findings

### Current State (What's Working Well)
- Help tips toggle with value-range descriptions on all controls
- Pipeline stages accordion explaining progress bar behavior
- Progress bar with tqdm integration shows real-time denoising steps
- Clear error messages for OOM and missing inputs
- Example galleries with one-click loading of presets
- Quick prompt lists for common subjects and lighting styles
- Auto-save to outputs/ with timestamps

### Current Gaps (UI/UX Issues Identified)

#### Navigation & Discoverability
1. **Two separate apps** - Users must choose between gradio_demo.py and gradio_demo_bg.py before launching. No way to switch modes without restarting.
2. **No onboarding** - New users see a complex interface with no guided workflow. The help tips help but aren't a tutorial.
3. **Advanced options hidden by default** - Good for reducing clutter, but scheduler (which affects quality significantly) is buried.

#### Input Workflow
4. **No seed randomization** - Users must manually type a new number to get variations. Every other SD interface has a dice/randomize button.
5. **No aspect ratio presets** - Users must manually set width and height. Most want standard ratios (portrait, landscape, square).
6. **No preset quality levels** - New users don't know what combination of steps/CFG/scale gives good results fast vs. high quality.
7. **Prompt building is manual** - Quick lists help but can't be combined easily. Subject + lighting style requires two clicks and the combination logic is fragile.

#### Output Workflow
8. **No before/after comparison** - After relighting, there's no easy way to compare input vs output side-by-side.
9. **No generation metadata on outputs** - Saved PNGs don't include the settings used. If a user finds a great result weeks later, they can't reproduce it.
10. **No history** - Each generation replaces the gallery. Previous results are only accessible via the filesystem.
11. **Gallery doesn't show which settings produced which image** - When generating multiple samples, all look similar with no way to tell them apart.

#### Performance & Feedback
12. **Relight button stays clickable during processing** - Can accidentally queue multiple generations.
13. **No estimated time remaining** - Progress bar shows percentage but not ETA.
14. **No cancel button** - Once started, generation must complete. Long generations (high steps, large size) can't be interrupted.

#### Accessibility
15. **No mobile-responsive layout** - Two-column layout doesn't adapt to narrow screens.
16. **No keyboard shortcuts** - Power users can't trigger generation or switch settings via keyboard.

## Technical Decisions
| Decision | Rationale |
|----------|-----------|
| JS-based info toggle instead of CSS | `[data-testid='block-info']` is the label, info text is a sibling div |
| `track_tqdm=True` for progress | `callback_on_step_end` doesn't flush to Gradio SSE frontend |
| DDIM default on MPS | DPMSolverMultistepScheduler has off-by-one indexing bug on MPS |

## Issues Encountered
| Issue | Resolution |
|-------|------------|
| CSS `.info` class targets toasts, not info text | Switched to JS targeting `nextElementSibling` of `[data-testid='block-info']` |
| `callback_on_step_end` progress stuck at 10% | Switched to `gr.Progress(track_tqdm=True)` |

## Resources
- Gradio 6.5.1 docs: component info text rendered as `<span data-testid="block-info">` (label) + sibling `<div>` (info)
- TECHNICAL.md: Architecture, pipeline flow, improvement paths
- CLAUDE.md: Development reference and gotchas

---
*Update this file after every 2 view/browser/search operations*
