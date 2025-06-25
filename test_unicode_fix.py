#!/usr/bin/env python3
"""
Test Unicode/emoji fixes for Windows
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def test_unicode_fixes():
    """Test that Unicode handling works"""
    print("🧪 Testing Unicode fixes...")
    
    try:
        from utils import clean_unicode_for_console, create_safe_logger, setup_logging
        
        # Test emoji cleaning
        test_messages = [
            "🚀 Starting Coheron",
            "🧬 Evolution generation 1",
            "📊 Best score: 0.85",
            "🏆 Breakthroughs: 2",
            "💾 Results saved",
            "✅ Success!"
        ]
        
        print("✅ Testing emoji cleaning:")
        for msg in test_messages:
            cleaned = clean_unicode_for_console(msg)
            print(f"   Original: {repr(msg)}")
            print(f"   Cleaned:  {cleaned}")
        
        # Test safe logger
        print("\n✅ Testing safe logger:")
        base_logger = setup_logging()
        safe_logger = create_safe_logger(base_logger)
        
        safe_logger.info("🧪 This is a test with emojis")
        safe_logger.info("Regular ASCII message")
        
        print("✅ Safe logger test completed!")
        
        return True
        
    except Exception as e:
        print(f"❌ Unicode test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run the test"""
    print("Testing Unicode/Emoji fixes for Windows")
    print("=" * 50)
    
    success = test_unicode_fixes()
    
    print("\n" + "=" * 50)
    if success:
        print("✅ All Unicode fixes working!")
    else:
        print("❌ Unicode fixes need more work")
    
    return success

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)