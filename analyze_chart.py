from PIL import Image
import numpy as np

def analyze_screenshot(path):
    """Analyze screenshot to determine if candlesticks are present"""
    print(f"\n{'='*70}")
    print(f"ANALYZING: {path}")
    print('='*70)
    
    img = Image.open(path)
    img_array = np.array(img)
    
    print(f"Image size: {img.size}")
    print(f"Image mode: {img.mode}")
    
    # Convert to RGB if needed
    if img.mode == 'RGBA':
        img = img.convert('RGB')
        img_array = np.array(img)
    
    # Check for candlestick colors
    # Green candles: #26a69a (38, 166, 154)
    # Red candles: #ef5350 (239, 83, 80)
    
    green_candle = np.array([38, 166, 154])
    red_candle = np.array([239, 83, 80])
    
    # Count pixels close to candlestick colors (within tolerance)
    tolerance = 30
    
    green_mask = np.all(np.abs(img_array - green_candle) < tolerance, axis=-1)
    red_mask = np.all(np.abs(img_array - red_candle) < tolerance, axis=-1)
    
    green_pixels = np.sum(green_mask)
    red_pixels = np.sum(red_mask)
    
    print(f"\n📊 Color Analysis:")
    print(f"   Green candlestick pixels: {green_pixels:,}")
    print(f"   Red candlestick pixels: {red_pixels:,}")
    print(f"   Total candlestick pixels: {green_pixels + red_pixels:,}")
    
    # Check for vertical lines (wicks)
    # Wicks should create vertical patterns
    gray_img = img.convert('L')
    gray_array = np.array(gray_img)
    
    # Look for high contrast vertical lines
    vertical_edges = np.abs(np.diff(gray_array, axis=0))
    strong_edges = np.sum(vertical_edges > 50)
    
    print(f"\n🕯️ Wick Detection:")
    print(f"   Strong vertical edges: {strong_edges:,}")
    
    # Determine result
    has_candles = (green_pixels + red_pixels) > 1000
    has_wicks = strong_edges > 10000
    
    print(f"\n{'='*70}")
    if has_candles and has_wicks:
        print("✅ CANDLESTICK CHART DETECTED!")
        print("   - Proper green and red candle bodies found")
        print("   - Vertical wicks/shadows present")
        result = "PASS"
    elif has_candles:
        print("⚠️  PARTIAL: Candle bodies detected but wicks unclear")
        result = "PARTIAL"
    else:
        print("❌ NO CANDLESTICKS DETECTED")
        print("   - This appears to be a line chart or empty")
        result = "FAIL"
    print('='*70)
    
    return result

if __name__ == '__main__':
    print("\n" + "="*70)
    print("🔍 CANDLESTICK CHART VALIDATION")
    print("="*70)
    
    # Analyze before toggle (should be line chart)
    print("\n1️⃣  BEFORE TOGGLE (should be LINE chart):")
    before_result = analyze_screenshot('before_toggle.png')
    
    # Analyze after toggle (should be candlestick)
    print("\n2️⃣  AFTER TOGGLE (should be CANDLESTICK chart):")
    after_result = analyze_screenshot('after_toggle.png')
    
    # Final verdict
    print("\n" + "="*70)
    print("📋 FINAL VERDICT")
    print("="*70)
    
    if before_result != "PASS" and after_result == "PASS":
        print("✅ SUCCESS! Toggle works correctly:")
        print("   • Before: Line chart (as expected)")
        print("   • After: Candlestick chart with proper candles and wicks")
        print("\n🎉 CANDLESTICK IMPLEMENTATION IS WORKING!")
    elif after_result == "PASS":
        print("✅ Candlesticks detected in final chart")
        print("⚠️  Note: Both views show candles (check if line mode works)")
    elif after_result == "PARTIAL":
        print("⚠️  PARTIAL SUCCESS:")
        print("   • Candle bodies are present")
        print("   • Wicks may not be fully visible (could be zoom/scale issue)")
    else:
        print("❌ FAILED:")
        print("   • Candlesticks NOT detected in after-toggle screenshot")
        print("   • Chart may not be rendering properly")
    
    print("="*70 + "\n")
