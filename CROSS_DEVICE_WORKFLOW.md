# Cross-Device Workflow Guide

How to resume Claude Code sessions across different devices using GitHub.

## 🔄 Session Continuity Across Devices

### What Transfers via GitHub ✅

**1. Code & Project Files** (In your repo)
- `CLAUDE.md` - Project instructions that guide any Claude session
- `SESSION_NOTES.md` - Latest session summary and implementation details
- `TECHNICAL.md` - Deep technical documentation
- All code changes and commits

**2. Context Files**
- Session summaries
- Progress logs
- Architecture documentation

### What Doesn't Transfer ❌

- **Conversation history** - Stored locally on your device
- **Session state** - Active context window content
- **Local memory files** - `.claude/` directory is device-specific

## 📱 How to Resume Work on Another Device

### Step 1: Pull Latest Changes

If you haven't cloned the repo yet:
```bash
git clone https://github.com/artfulreflections/IC-Light-Apple-Silicon.git
cd IC-Light-Apple-Silicon
```

If you already have the repo:
```bash
cd IC-Light-Apple-Silicon
git pull origin main
```

### Step 2: Open in Claude Code

```bash
claude-code .
```

Or open your IDE and start Claude Code in the project directory.

### Step 3: Give Context to New Session

When starting a new Claude Code session, provide context by saying:

> "Read SESSION_NOTES.md to understand what was implemented in the last session. I want to continue working on this project."

The new Claude Code session will have:
- ✅ **All code changes** (preview button, cancel button, preprocessing)
- ✅ **Project instructions** (CLAUDE.md auto-loads)
- ✅ **Session context** (SESSION_NOTES.md has everything we did)
- ✅ **Architecture knowledge** (TECHNICAL.md, README.md)

## 📚 What Each File Provides

| File | Purpose | Auto-loads? | When to Read |
|------|---------|-------------|--------------|
| `CLAUDE.md` | Project instructions, patterns, architecture | ✅ Yes | Automatic |
| `SESSION_NOTES.md` | Latest session summary, next steps | ❌ No | When resuming work |
| `TECHNICAL.md` | Deep technical documentation | ❌ No | When need details |
| `CROSS_DEVICE_WORKFLOW.md` | This guide | ❌ No | When switching devices |
| Code files | Actual implementation | ✅ Yes | Automatic |

## 🎯 Best Practice Workflow

### Before Ending Session on Device A

1. **Update session notes** with current progress:
   ```bash
   # Edit SESSION_NOTES.md with what you're working on
   ```

2. **Commit and push**:
   ```bash
   git add SESSION_NOTES.md
   git commit -m "Update session notes: <what you did>"
   git push origin main
   ```

3. **Note any in-progress work** in SESSION_NOTES.md

### When Starting on Device B

1. **Pull latest changes**:
   ```bash
   git pull origin main
   ```

2. **Open in Claude Code**:
   ```bash
   claude-code .
   ```

3. **Provide context to Claude**:
   > "Read SESSION_NOTES.md and continue where we left off"

4. **Verify environment**:
   ```bash
   # Check Python version
   python --version  # Should be 3.12

   # Activate virtual environment
   source .venv/bin/activate  # macOS/Linux
   .venv\Scripts\activate     # Windows

   # Verify dependencies
   pip list
   ```

## 📝 Session Notes Template

When updating `SESSION_NOTES.md`, use this format:

```markdown
## Session [Date]

### What I Implemented
- Feature 1: Description
- Feature 2: Description

### What I'm Working On
- Current task in progress
- Next immediate steps

### Known Issues
- Any bugs or problems to address

### Next Steps
- [ ] Task 1
- [ ] Task 2
```

## 🚀 Quick Commands Reference

### Git Operations
```bash
# Check status
git status

# Pull latest
git pull origin main

# Commit changes
git add .
git commit -m "Description"
git push origin main

# View recent commits
git log --oneline -5
```

### Python Environment
```bash
# Activate venv
source .venv/bin/activate

# Run tests
pytest tests/

# Run text-conditioned demo
python gradio_demo.py

# Run background-conditioned demo
python gradio_demo_bg.py
```

### Project-Specific
```bash
# Run with custom settings
python gradio_demo.py --host 0.0.0.0 --port 7860 --model-dir ./models

# Check syntax
python -m py_compile gradio_demo.py gradio_demo_bg.py utils.py

# Run linter
ruff check .
```

## 💡 Tips for Seamless Transitions

1. **Commit frequently** - Small, focused commits make it easier to understand what changed

2. **Keep SESSION_NOTES.md current** - Update it throughout your session, not just at the end

3. **Use descriptive commit messages** - Future you will thank present you

4. **Tag important milestones** - Use git tags for major feature completions:
   ```bash
   git tag -a v1.0-preview-feature -m "Preview and cancel buttons implemented"
   git push origin v1.0-preview-feature
   ```

5. **Document blockers** - If you hit a wall, document it in SESSION_NOTES.md so you can get help or think about it fresh

## 🔧 Troubleshooting

### "Claude doesn't remember what I was doing"
✅ **Solution**: Ask Claude to read SESSION_NOTES.md explicitly

### "Code is different than I expected"
✅ **Solution**: Check git log to see what commits were made:
```bash
git log --oneline --graph --all -10
```

### "Environment issues on new device"
✅ **Solution**: Verify Python version and reinstall dependencies:
```bash
python --version  # Should be 3.12.x
pip install -e .
```

### "Can't find project context"
✅ **Solution**: All project context files are in the repo root:
- CLAUDE.md
- SESSION_NOTES.md
- TECHNICAL.md
- README.md

## 📧 Alternative: Conversation Export

While SESSION_NOTES.md is the recommended approach, you could also:

1. Export conversation transcript (if Claude Code supports it)
2. Save as `conversations/session-YYYY-MM-DD.md`
3. Commit to repo (but these get large quickly)

**Recommendation**: Stick with SESSION_NOTES.md for cleaner, more maintainable documentation.

---

## Summary

**Your session is now fully portable!** 🚀

The combination of:
- Git version control
- CLAUDE.md for project instructions
- SESSION_NOTES.md for session context
- Comprehensive documentation

Means you can seamlessly continue work on any device with full context.
