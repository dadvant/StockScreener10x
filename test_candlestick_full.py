from playwright.sync_api import sync_playwright
import time

def test_candlestick():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)  # Show browser
        context = browser.new_context()
        page = context.new_page()

        def snap_canvas(path: str):
            try:
                canvas = page.query_selector('#priceChart')
                if not canvas:
                    print(f"✗ Canvas not found for {path}")
                    return
                box = canvas.bounding_box()
                if not box:
                    print(f"✗ Canvas bounding box unavailable for {path}")
                    return
                page.screenshot(path=path, clip=box)
                print(f"📸 Canvas screenshot saved: {path}")
            except Exception as e:
                print(f"✗ Canvas screenshot error for {path}: {e}")
        
        # Capture console messages
        console_logs = []
        def handle_console(msg):
            try:
                console_logs.append(f"{msg.type}: {msg.text}")
            except:
                pass
        page.on('console', handle_console)
        
        # Capture errors
        errors = []
        page.on('pageerror', lambda exc: errors.append(str(exc)))
        
        print("🔍 Loading main page...")
        page.goto('http://127.0.0.1:5000/')
        page.wait_for_load_state('networkidle')
        time.sleep(2)
        
        # Click TEST AAPL button
        print("🔘 Clicking TEST AAPL button...")
        page.click('#testAaplBtn')
        time.sleep(3)
        
        # Wait for detail view
        page.wait_for_selector('#detailView', state='visible', timeout=5000)
        print("✓ Detail view visible")
        
        # Check if chart canvas exists
        canvas = page.query_selector('#priceChart')
        if canvas:
            print("✓ Canvas element exists")
        else:
            print("✗ Canvas element NOT FOUND")
            
        # Check initial chart state
        initial_chart_type = page.evaluate("""() => {
            return window.currentChartType || 'unknown';
        }""")
        print(f"📊 Initial chart type: {initial_chart_type}")
        
        # Check if Chart.js is loaded
        chart_exists = page.evaluate("""() => {
            return typeof Chart !== 'undefined';
        }""")
        print(f"📈 Chart.js loaded: {chart_exists}")
        
        # Check if financial controller is available
        has_financial = page.evaluate("""() => {
            return !!(Chart?.registry?.getController?.('candlestick'));
        }""")
        print(f"🕯️ Financial controller available: {has_financial}")
        
        # Take screenshot before toggle
        page.screenshot(path='before_toggle.png')
        page.wait_for_timeout(800)
        snap_canvas('before_toggle_chart.png')
        
        # Click candlestick toggle
        print("🔘 Clicking candlestick toggle...")
        toggle = page.query_selector('#chartToggle')
        if toggle:
            visible = toggle.is_visible()
            print(f"   Toggle visible: {visible}")
            if visible:
                toggle.click()
                time.sleep(2)
                print("✓ Toggle clicked")
            else:
                print("✗ Toggle not visible")
        else:
            print("✗ Toggle button not found")
        
        # Check chart type after toggle
        after_chart_type = page.evaluate("""() => {
            return window.currentChartType || 'unknown';
        }""")
        print(f"📊 After toggle chart type: {after_chart_type}")
        
        # Check if chart instance exists
        chart_info = page.evaluate("""() => {
            const canvas = document.getElementById('priceChart');
            if (!canvas) return { error: 'No canvas' };
            
            const chart = Chart.getChart('priceChart');
            if (!chart) return { error: 'No Chart instance' };
            
            return {
                type: chart.config.type,
                datasets: chart.data.datasets.length,
                datasetTypes: chart.data.datasets.map(d => d.type || chart.config.type),
                dataPoints: chart.data.datasets[0]?.data?.length || 0
            };
        }""")
        print(f"📊 Chart instance info: {chart_info}")
        
        # Take screenshot after toggle
        page.screenshot(path='after_toggle.png')
        page.wait_for_timeout(800)
        snap_canvas('after_toggle_chart.png')
        
        # Get the canvas rendering
        is_blank = page.evaluate("""() => {
            const canvas = document.getElementById('priceChart');
            if (!canvas) return true;
            
            const ctx = canvas.getContext('2d');
            const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
            const data = imageData.data;
            
            // Check if all pixels are the same (blank canvas)
            let allSame = true;
            const firstPixel = [data[0], data[1], data[2], data[3]];
            for (let i = 0; i < data.length; i += 4) {
                if (data[i] !== firstPixel[0] || data[i+1] !== firstPixel[1] || 
                    data[i+2] !== firstPixel[2] || data[i+3] !== firstPixel[3]) {
                    allSame = false;
                    break;
                }
            }
            return allSame;
        }""")
        print(f"🎨 Canvas is blank: {is_blank}")
        
        # Print relevant console logs
        print("\n📝 Console logs:")
        for log in console_logs[-20:]:
            if 'candlestick' in log.lower() or 'chart' in log.lower() or 'financial' in log.lower():
                print(f"   {log}")
        
        # Print errors
        if errors:
            print("\n❌ Errors:")
            for error in errors:
                print(f"   {error}")
        else:
            print("\n✓ No JavaScript errors")
        
        print("\n✅ Test complete! Check screenshots.")
        time.sleep(5)  # Keep browser open for manual inspection
        
        browser.close()

if __name__ == '__main__':
    test_candlestick()
