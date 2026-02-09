#!/usr/bin/env python3
"""Quick test for seed management functions"""

import os
import shutil
import tempfile

from utils import get_favorite_seeds_choices, load_favorite_seeds, save_favorite_seed

def test_seed_management():
    # Create a temporary directory for testing
    temp_dir = tempfile.mkdtemp()

    try:
        print(f"Testing in: {temp_dir}")

        # Test 1: Save a favorite seed
        print("\n1. Testing save_favorite_seed...")
        success = save_favorite_seed(temp_dir, 12345, "Test seed 1")
        assert success, "Failed to save seed"
        print("✅ Save seed succeeded")

        # Test 2: Load favorite seeds
        print("\n2. Testing load_favorite_seeds...")
        favorites = load_favorite_seeds(temp_dir)
        assert len(favorites) == 1, f"Expected 1 favorite, got {len(favorites)}"
        assert favorites[0]['seed'] == 12345, f"Expected seed 12345, got {favorites[0]['seed']}"
        assert favorites[0]['label'] == "Test seed 1", f"Expected label 'Test seed 1', got {favorites[0]['label']}"
        print(f"✅ Load seeds succeeded: {favorites}")

        # Test 3: Save another seed
        print("\n3. Testing save another seed...")
        success = save_favorite_seed(temp_dir, 67890, "Test seed 2", settings={'steps': 25, 'cfg': 2.0})
        assert success, "Failed to save second seed"
        print("✅ Save second seed succeeded")

        # Test 4: Load multiple seeds
        print("\n4. Testing load multiple seeds...")
        favorites = load_favorite_seeds(temp_dir)
        assert len(favorites) == 2, f"Expected 2 favorites, got {len(favorites)}"
        print(f"✅ Load multiple seeds succeeded: {len(favorites)} seeds")

        # Test 5: Update existing seed
        print("\n5. Testing update existing seed...")
        success = save_favorite_seed(temp_dir, 12345, "Updated test seed 1")
        assert success, "Failed to update seed"
        favorites = load_favorite_seeds(temp_dir)
        assert len(favorites) == 2, f"Expected 2 favorites (not 3), got {len(favorites)}"
        found = False
        for fav in favorites:
            if fav['seed'] == 12345:
                assert fav['label'] == "Updated test seed 1", f"Label not updated: {fav['label']}"
                found = True
                break
        assert found, "Updated seed not found"
        print("✅ Update seed succeeded")

        # Test 6: Get choices for Gradio dropdown
        print("\n6. Testing get_favorite_seeds_choices...")
        choices = get_favorite_seeds_choices(temp_dir)
        assert len(choices) == 2, f"Expected 2 choices, got {len(choices)}"
        assert choices[0][1] == 12345 or choices[1][1] == 12345, "Seed 12345 not in choices"
        print(f"✅ Get choices succeeded: {choices}")

        # Test 7: Test with empty directory
        print("\n7. Testing with empty directory...")
        empty_dir = os.path.join(temp_dir, "empty")
        os.makedirs(empty_dir, exist_ok=True)
        favorites = load_favorite_seeds(empty_dir)
        assert len(favorites) == 0, f"Expected 0 favorites in empty dir, got {len(favorites)}"
        choices = get_favorite_seeds_choices(empty_dir)
        assert len(choices) == 0, f"Expected 0 choices in empty dir, got {len(choices)}"
        print("✅ Empty directory test passed")

        print("\n" + "="*50)
        print("✨ All tests passed!")
        print("="*50)

    finally:
        # Clean up
        shutil.rmtree(temp_dir)
        print(f"\nCleaned up temp directory: {temp_dir}")

if __name__ == "__main__":
    test_seed_management()
