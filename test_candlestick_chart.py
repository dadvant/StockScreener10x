#!/usr/bin/env python3
import json
import requests
from pathlib import Path

# Get AAPL data from API
print("=" * 60)
print("CANDLESTICK CHART VALIDATION TEST")
print("=" * 60)

try:
    response = requests.get('http://localhost:5000/api/stock/AAPL', timeout=10)
    if response.status_code != 200:
        print(f"✗ API returned {response.status_code}")
        exit(1)
    
    data = response.json()
    if 'data' not in data:
        print("✗ No 'data' field in response")
        exit(1)
    
    stock_data = data['data']
    
    # Validate OHLC data
    print("\n✓ API Response OK")
    print(f"  • open: {len(stock_data.get('open', []))} values")
    print(f"  • high: {len(stock_data.get('high', []))} values")
    print(f"  • low: {len(stock_data.get('low', []))} values")
    print(f"  • close: {len(stock_data.get('close', []))} values")
    print(f"  • dates: {len(stock_data.get('dates', []))} dates")
    print(f"  • ma_50: {len(stock_data.get('ma_50', []))} values")
    print(f"  • ma_200: {len(stock_data.get('ma_200', []))} values")
    
    # Check for data completeness
    open_data = stock_data.get('open', [])
    high_data = stock_data.get('high', [])
    low_data = stock_data.get('low', [])
    close_data = stock_data.get('close', [])
    
    if not all([open_data, high_data, low_data, close_data]):
        print("\n✗ Missing OHLC data arrays")
        exit(1)
    
    # Validate candlestick rules
    print("\n✓ OHLC Data Validation:")
    issues = 0
    for i in range(min(5, len(open_data))):
        o = open_data[i]
        h = high_data[i]
        l = low_data[i]
        c = close_data[i]
        
        # Check rules: high >= max(o,c) and low <= min(o,c)
        if not (h >= o and h >= c):
            print(f"  ✗ Bar {i}: high ({h}) not >= open ({o}) and close ({c})")
            issues += 1
        if not (l <= o and l <= c):
            print(f"  ✗ Bar {i}: low ({l}) not <= open ({o}) and close ({c})")
            issues += 1
        
        if issues == 0:
            bullish = "🟢" if c >= o else "🔴"
            print(f"  ✓ Bar {i}: O={o:.2f} H={h:.2f} L={l:.2f} C={c:.2f} {bullish}")
    
    if issues > 0:
        print(f"\n✗ {issues} validation errors")
        exit(1)
    
    # Generate test HTML
    print("\n✓ Generating test HTML...")
    
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Candlestick Chart Test</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@3.9.1/dist/chart.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-luxon@1.2.0"></script>
    <style>
        body {{ font-family: Arial; background: #1a1f2e; color: #fff; padding: 20px; }}
        canvas {{ max-width: 100%; max-height: 500px; }}
        #container {{ max-width: 1400px; margin: 0 auto; }}
        h1 {{ text-align: center; }}
    </style>
</head>
<body>
    <div id="container">
        <h1>🕯️ Candlestick Chart - AAPL (3Y)</h1>
        <canvas id="myChart"></canvas>
    </div>

    <script>
        const chartData = {json.dumps(stock_data)};
        
        const ctx = document.getElementById('myChart').getContext('2d');
        const labels = chartData.dates;
        const open = chartData.open;
        const high = chartData.high;
        const low = chartData.low;
        const close = chartData.close;
        const ma50 = chartData.ma_50;
        const ma200 = chartData.ma_200;
        
        console.log('Data loaded:');
        console.log('  bars:', labels.length);
        console.log('  open:', open.length, '  high:', high.length);
        console.log('  low:', low.length, '  close:', close.length);
        
        // Build candlestick datasets
        const datasets = [];
        
        // Wicks (scatter)
        const wickData = labels.map((date, i) => ({{
            x: date,
            y: (high[i] + low[i]) / 2,
            h: high[i],
            l: low[i]
        }}));
        
        datasets.push({{
            label: 'Wicks',
            type: 'scatter',
            data: wickData,
            borderColor: (ctx) => {{
                if (ctx.dataIndex === undefined) return '#999';
                const idx = ctx.dataIndex;
                return close[idx] >= open[idx] ? '#26a69a' : '#ef5350';
            }},
            borderWidth: 1,
            pointRadius: 0,
            showLine: false
        }});
        
        // Bodies (bars)
        const bodyData = labels.map((date, i) => ({{
            x: date,
            y: Math.max(open[i], close[i]),
            base: Math.min(open[i], close[i])
        }}));
        
        const colors = close.map((c, i) => c >= open[i] ? '#26a69a' : '#ef5350');
        
        datasets.push({{
            label: 'OHLC',
            type: 'bar',
            data: bodyData,
            backgroundColor: colors,
            borderColor: colors,
            borderWidth: 1,
            barPercentage: 0.6,
            categoryPercentage: 0.8
        }});
        
        // MA50
        if (ma50 && ma50.length > 0) {{
            datasets.push({{
                label: 'MA 50',
                data: labels.map((date, i) => ({{x: date, y: ma50[i]}})),
                borderColor: '#ffb300',
                borderWidth: 2,
                borderDash: [5, 5],
                fill: false,
                pointRadius: 0,
                type: 'line'
            }}));
        }}
        
        // MA200
        if (ma200 && ma200.length > 0) {{
            datasets.push({{
                label: 'MA 200',
                data: labels.map((date, i) => ({{x: date, y: ma200[i]}})),
                borderColor: '#e53935',
                borderWidth: 2,
                borderDash: [5, 5],
                fill: false,
                pointRadius: 0,
                type: 'line'
            }}));
        }}
        
        const chart = new Chart(ctx, {{
            type: 'line',
            data: {{ labels: labels, datasets: datasets }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{ display: true, labels: {{ color: '#e0e0e0' }} }},
                    tooltip: {{
                        backgroundColor: 'rgba(26, 31, 46, 0.95)',
                        titleColor: '#fff',
                        bodyColor: '#e0e0e0',
                        callbacks: {{
                            label: function(context) {{
                                if (context.dataset.label === 'OHLC') {{
                                    const idx = context.dataIndex;
                                    return `O:${{open[idx].toFixed(2)}} H:${{high[idx].toFixed(2)}} L:${{low[idx].toFixed(2)}} C:${{close[idx].toFixed(2)}}`;
                                }}
                                let label = context.dataset.label || '';
                                if (label) label += ': ';
                                if (context.parsed.y !== null) {{
                                    label += '$' + context.parsed.y.toFixed(2);
                                }}
                                return label;
                            }}
                        }}
                    }}
                }},
                scales: {{
                    x: {{ grid: {{ color: 'rgba(42, 63, 95, 0.3)' }}, ticks: {{ color: '#999' }} }},
                    y: {{ grid: {{ color: 'rgba(42, 63, 95, 0.3)' }}, ticks: {{ color: '#999' }} }}
                }}
            }}
        }});
        
        console.log('✓ Chart rendered');
    </script>
</body>
</html>"""
    
    # Write test HTML
    test_file = Path('test_candlestick_chart.html')
    test_file.write_text(html)
    print(f"  ✓ Created: {test_file}")
    
    print("\n" + "=" * 60)
    print("✓ VALIDATION COMPLETE - OPEN: test_candlestick_chart.html")
    print("=" * 60)
    
except Exception as e:
    print(f"✗ Error: {e}")
    exit(1)
