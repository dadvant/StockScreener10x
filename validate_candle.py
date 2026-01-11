#!/usr/bin/env python
import json
from pathlib import Path

print("="*60)
print("CANDLESTICK CHART TEST")
print("="*60)

# Test data
dates = [f"2025-01-{i:02d}" for i in range(1, 21)]
opens = [150.0 + i * 0.5 for i in range(20)]
highs = [o + 2.5 for o in opens]
lows = [o - 1.5 for o in opens]
closes = [150.5 + i * 0.7 for i in range(20)]
ma_50 = [150.0 + i * 0.6 for i in range(20)]
ma_200 = [149.5 + i * 0.4 for i in range(20)]

data = {
    'dates': dates,
    'open': opens,
    'high': highs,
    'low': lows,
    'close': closes,
    'ma_50': ma_50,
    'ma_200': ma_200
}

# Validate
print("\nValidation:")
for i in range(3):
    o, h, l, c = opens[i], highs[i], lows[i], closes[i]
    bull = "🟢" if c >= o else "🔴"
    print(f"  Bar {i}: O={o:.1f} H={h:.1f} L={l:.1f} C={c:.1f} {bull}")

html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Candlestick Test</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@3.9.1"></script>
    <style>
        body {{ font-family: Arial; background: #1a1f2e; color: #fff; padding: 20px; }}
        h1 {{ text-align: center; color: #fff; }}
        #chart {{ height: 600px; max-width: 1200px; margin: 0 auto; }}
    </style>
</head>
<body>
    <h1>🕯️ Candlestick Chart Validation</h1>
    <canvas id="chart"></canvas>
    <script>
        const data = {json.dumps(data)};
        const o = data.open;
        const h = data.high;
        const l = data.low;
        const c = data.close;
        const d = data.dates;
        const m50 = data.ma_50;
        const m200 = data.ma_200;
        
        const datasets = [];
        
        // Wicks
        datasets.push({{
            label: 'Wicks',
            type: 'scatter',
            data: d.map((x, i) => ({{ x, y: (h[i] + l[i])/2 }})),
            borderColor: d.map((_, i) => c[i] >= o[i] ? '#26a69a' : '#ef5350'),
            borderWidth: 2,
            pointRadius: 0
        }});
        
        // Bodies
        datasets.push({{
            label: 'OHLC',
            type: 'bar',
            data: d.map((x, i) => ({{ x, y: Math.max(o[i], c[i]), base: Math.min(o[i], c[i]) }})),
            backgroundColor: c.map((v, i) => v >= o[i] ? '#26a69a' : '#ef5350'),
            borderColor: c.map((v, i) => v >= o[i] ? '#26a69a' : '#ef5350'),
            barPercentage: 0.6
        }});
        
        // MA50
        datasets.push({{
            label: 'MA 50',
            type: 'line',
            data: d.map((x, i) => ({{ x, y: m50[i] }})),
            borderColor: '#ffb300',
            borderDash: [5, 5],
            fill: false,
            borderWidth: 2,
            pointRadius: 0
        }});
        
        // MA200
        datasets.push({{
            label: 'MA 200',
            type: 'line',
            data: d.map((x, i) => ({{ x, y: m200[i] }})),
            borderColor: '#e53935',
            borderDash: [5, 5],
            fill: false,
            borderWidth: 2,
            pointRadius: 0
        }});
        
        new Chart(document.getElementById('chart'), {{
            type: 'line',
            data: {{ labels: d, datasets: datasets }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{ labels: {{ color: '#e0e0e0' }} }},
                    tooltip: {{ backgroundColor: 'rgba(0,0,0,0.8)', titleColor: '#fff', bodyColor: '#e0e0e0' }}
                }},
                scales: {{
                    x: {{ ticks: {{ color: '#999' }} }},
                    y: {{ ticks: {{ color: '#999' }} }}
                }}
            }}
        }});
    </script>
</body>
</html>"""

Path('test_candlestick_chart.html').write_text(html)
print("\n✓ Generated: test_candlestick_chart.html")
print("✓ Expected: Green/red bars with MA overlays")
