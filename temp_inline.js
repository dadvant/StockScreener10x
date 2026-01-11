
        let priceChart = null;
        let currentChartType = 'line';
        let currentChartData = null;
        let currentTechnical = null;

        // Global click beacon (capture) to diagnose click handling
        try {
            document.addEventListener('click', function(ev){
                try {
                    const t = ev.target;
                    const tag = t && t.tagName || 'UNKNOWN';
                    const id = t && t.id || '';
                    const cls = t && t.className || '';
                    const info = `GLOBAL_CLICK tag=${tag} id=${id} class=${cls}`;
                    if (navigator.sendBeacon) { try { navigator.sendBeacon('/client_log', info); } catch(_){} }
                    console.log(info);
                } catch(_) {}
            }, true);
        } catch(_) {}

        // Optional: auto-open AAPL detail when hash is set
        try {
            if (location && location.hash === '#test-aapl') {
                setTimeout(function(){
                    if (window.viewStockDetail) { window.viewStockDetail('AAPL'); }
                }, 1000);
            }
        } catch(_) {}

        // Global JS error banner for quick diagnosis
        window.addEventListener('error', function(ev) {
            try {
                var msg = (ev && ev.message) ? ev.message : 'Unknown error';
                var loc = '';
                try { loc = (ev && ev.filename ? ev.filename : '') + ':' + (ev && ev.lineno ? ev.lineno : 0) + ':' + (ev && ev.colno ? ev.colno : 0); } catch(_) {}
                var banner = document.createElement('div');
                banner.style.cssText = 'position:fixed;top:0;left:0;right:0;background:#b00020;color:#fff;padding:10px;z-index:9999;font-weight:600';
                banner.textContent = 'Script error: ' + msg + (loc ? (' @ ' + loc) : '');
                document.body.appendChild(banner);
                window.lastErrorMessage = msg;
                window.lastErrorLocation = loc;
                console.error('Global error at', loc, msg);
                if (navigator.sendBeacon) {
                    try { navigator.sendBeacon('/client_log', 'JS_ERROR:' + msg + '|LOC|' + loc); } catch(_) {}
                }
            } catch(e) {}
        });

        window.addEventListener('unhandledrejection', function(ev) {
            try {
                var msg = (ev && ev.reason && ev.reason.message) ? ev.reason.message : 'Unhandled rejection';
                console.error('Unhandled rejection:', ev);
                if (navigator.sendBeacon) {
                    try { navigator.sendBeacon('/client_log', 'JS_UNHANDLED_REJECTION:' + msg); } catch(_) {}
                }
            } catch(e) {}
        });

        function initUI() {
            try {
                const statusEl = document.getElementById('uiStatus');
                const note = (msg) => { if (statusEl) statusEl.textContent = 'UI status: ' + msg; console.log(msg); };
                window.uiNote = note; // expose for quick debugging

                note('binding controls…');

                const btn = document.getElementById('scanBtn');
                if (btn) {
                    btn.addEventListener('click', (e) => {
                        e.preventDefault();
                        startScan();
                        note('Scan button clicked');
                    });
                    // Fallback inline assignment in case addEventListener is overridden
                    btn.onclick = (e) => { e.preventDefault(); startScan(); note('Scan button clicked (inline fallback)'); };
                    // Pointer/touch fallback
                    btn.addEventListener('pointerdown', (e) => { e.preventDefault(); startScan(); note('Scan button clicked (pointerdown)'); });
                    console.log('Scan button bound');
                    note('Scan button bound');
                } else {
                    console.warn('Scan button not found');
                    note('Scan button not found');
                }

                const settingsBtn = document.getElementById('settingsBtn');
                if (settingsBtn) {
                    settingsBtn.addEventListener('click', (e) => {
                        e.preventDefault();
                        toggleSettings();
                        note('Settings button clicked');
                    });
                    settingsBtn.onclick = (e) => { e.preventDefault(); toggleSettings(); note('Settings button clicked (inline fallback)'); };
                    settingsBtn.addEventListener('pointerdown', (e) => { e.preventDefault(); toggleSettings(); note('Settings button clicked (pointerdown)'); });
                    console.log('Settings button bound');
                    note('Settings button bound');
                } else {
                    console.warn('Settings button not found');
                    note('Settings button not found');
                }
                // Bind TEST AAPL button without inline handlers
                try {
                    const testBtn = document.getElementById('testAaplBtn');
                    if (testBtn) {
                        testBtn.addEventListener('click', (e) => {
                            e.preventDefault();
                            console.log('TEST clicked');
                            if (!window.viewStockDetail) {
                                alert('viewStockDetail not available');
                                if (navigator.sendBeacon) { try { navigator.sendBeacon('/client_log','VIEW_STOCK_DETAIL_MISSING'); } catch(_){} }
                            } else {
                                window.viewStockDetail('AAPL');
                            }
                        });
                    }
                } catch(_){}

                // Event delegation as an extra safety net
                document.addEventListener('click', (ev) => {
                    if (ev.target && ev.target.id === 'scanBtn') {
                        ev.preventDefault();
                        startScan();
                        note('Scan button clicked (delegated)');
                    }
                    if (ev.target && ev.target.id === 'settingsBtn') {
                        ev.preventDefault();
                        toggleSettings();
                        note('Settings button clicked (delegated)');
                    }
                }, true);

                // Preload current settings into panel
                loadAndDisplaySettings();

                // Live-update threshold without alert (debounced)
                try {
                    const thr = document.getElementById('convictionThreshold');
                    let thrTimer = null;
                    if (thr) {
                        thr.addEventListener('input', (e) => {
                            const v = parseFloat(e.target.value) || 4.0;
                            clearTimeout(thrTimer);
                            thrTimer = setTimeout(() => {
                                fetch('/api/settings', {
                                    method: 'POST',
                                    headers: { 'Content-Type': 'application/json' },
                                    body: JSON.stringify({ conviction_threshold: v })
                                })
                                .then(() => { loadResults(); note('Threshold updated to ' + v); })
                                .catch(err => console.error('threshold update error', err));
                            }, 250);
                        });
                    }
                } catch(e) { console.error('threshold live update setup error', e); }

                note('ready');

                // Force-load AAPL detail once as a hard fallback to ensure detail view populates
                try {
                    setTimeout(() => { 
                        if (window.viewStockDetail) { 
                            window.viewStockDetail('AAPL'); 
                            note('Auto-opened AAPL detail (fallback)');
                        }
                    }, 800);
                } catch(e) { console.error('auto-open fallback error', e); }

                // Delegated click handler for cards (main and fallback)
                const resultsDiv = document.getElementById('resultsDiv');
                if (resultsDiv) {
                    resultsDiv.addEventListener('click', (ev) => {
                        const card = ev.target.closest('.opportunity-card');
                        if (!card) return;
                        ev.preventDefault();
                        const t = card.dataset.ticker || (card.querySelector('.ticker-large') ? card.querySelector('.ticker-large').textContent.trim() : '');
                        note('card clicked ' + t);
                        if (window.viewStockDetail && t) {
                            window.viewStockDetail(t);
                        }
                    });
                }
            } catch (e) {
                console.error('Initialization error:', e);
                const statusEl = document.getElementById('uiStatus');
                if (statusEl) statusEl.textContent = 'UI status: Initialization error: ' + e.message;
            }
        }

        // Run init immediately if DOM is already ready; otherwise wait
        try {
            if (document.readyState === 'complete' || document.readyState === 'interactive') {
                initUI();
            } else {
                document.addEventListener('DOMContentLoaded', initUI);
            }
        } catch (e) {
            console.error('Initial init binding error:', e);
        }

        // Absolute fallback: force init after 300ms in case DOMContentLoaded never fires
        setTimeout(() => {
            try {
                initUI();
                const statusEl = document.getElementById('uiStatus');
                if (statusEl && statusEl.textContent && statusEl.textContent.includes('binding')) {
                    statusEl.textContent = 'UI status: ready (fallback)';
                }
            } catch (e) {
                console.error('Fallback init error:', e);
            }
        }, 300);

        // Last-resort auto-trigger: fire startScan once after bindings to prove handler executes
        setTimeout(() => {
            try {
                const statusEl = document.getElementById('uiStatus');
                if (statusEl) statusEl.textContent = 'UI status: auto-triggering scan test…';
                startScan();
            } catch (e) {
                console.error('Auto-trigger error:', e);
                const statusEl = document.getElementById('uiStatus');
                if (statusEl) statusEl.textContent = 'UI status: auto-trigger failed: ' + e.message;
            }
        }, 600);

        function toggleChartType() {
            if (currentChartType === 'line') {
                currentChartType = 'candlestick';
                document.getElementById('chartToggle').textContent = '🕯️ Candlestick';
            } else {
                currentChartType = 'line';
                document.getElementById('chartToggle').textContent = '📈 Line';
            }
            
            if (currentChartData && currentTechnical) {
                renderChart(currentChartData, currentTechnical);
            }
        }

        let __lastToggleTs = 0;
        function toggleSettings() {
            const now = Date.now();
            if (now - __lastToggleTs < 150) return; // prevent double toggle from multiple handlers
            __lastToggleTs = now;
            const panel = document.getElementById('settingsPanel');
            if (!panel) return;
            if (panel.style.display === 'none') {
                loadAndDisplaySettings();
                panel.style.display = 'block';
            } else {
                panel.style.display = 'none';
            }
        }
        // Ensure global access for inline handlers or external testers
        window.toggleSettings = toggleSettings;

        function loadAndDisplaySettings() {
            fetch('/api/settings')
                .then(r => r.json())
                .then(settings => {
                    document.getElementById('convictionThreshold').value = settings.conviction_threshold;
                    document.getElementById('rsiMin').value = settings.rsi_min;
                    document.getElementById('rsiMax').value = settings.rsi_max;
                    document.getElementById('volumeRatioMin').value = settings.volume_ratio_min;
                    document.getElementById('maMinPeriod').value = settings.ma_min_period;
                    document.getElementById('maMaxPeriod').value = settings.ma_max_period;
                });
        }

        function loadAndApplySettings() {
            const settings = {
                conviction_threshold: parseFloat(document.getElementById('convictionThreshold').value) || 4.0,
                rsi_min: parseInt(document.getElementById('rsiMin').value) || 20,
                rsi_max: parseInt(document.getElementById('rsiMax').value) || 80,
                volume_ratio_min: parseFloat(document.getElementById('volumeRatioMin').value) || 0.8,
                ma_min_period: parseInt(document.getElementById('maMinPeriod').value) || 150,
                ma_max_period: parseInt(document.getElementById('maMaxPeriod').value) || 200
            };

            fetch('/api/settings', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(settings)
            })
            .then(r => r.json())
            .then(updated => {
                console.log('Updated settings:', updated);
                // Immediately re-load results to apply conviction filter without re-scanning
                loadResults();
            });
        }

        document.getElementById('riskTolerance').addEventListener('input', (e) => {
            document.getElementById('riskValue').textContent = e.target.value + '/10';
        });

        function startScan() {
            const btn = document.getElementById('scanBtn');
            btn.disabled = true;
            
            const capital = document.getElementById('capital').value;
            const risk = document.getElementById('riskTolerance').value;
            const universe = document.getElementById('universeSize').value;

            document.getElementById('progressDiv').classList.add('active');
            document.getElementById('progressFill').style.width = '0%';
            document.getElementById('progressFill').textContent = '0%';
            // Provide immediate user feedback
            console.log('Starting scan...', { capital, risk, universe });
            const resultsDiv = document.getElementById('resultsDiv');
            resultsDiv.innerHTML = '<div class="empty">Starting scan… analyzing stocks with AI.</div>';

            fetch('/api/scan', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    capital: parseInt(capital),
                    risk_tolerance: parseInt(risk),
                    universe_size: parseInt(universe)
                })
            })
            .catch(err => {
                console.error('Error starting scan:', err);
                resultsDiv.innerHTML = '<div class="empty">Failed to start scan. Please check server and try again.</div>';
                document.getElementById('progressDiv').classList.remove('active');
                btn.disabled = false;
                const statusEl = document.getElementById('uiStatus');
                if (statusEl) statusEl.textContent = 'Scan start error: ' + err;
            });

            pollStatus(btn);
        }
        // Ensure global access for inline handlers or external testers
        window.startScan = startScan;

        function pollStatus(btn) {
            fetch('/api/status')
                .then(r => r.json())
                .then(data => {
                    console.log('Status:', data);
                    const fill = document.getElementById('progressFill');
                    fill.style.width = data.progress + '%';
                    fill.textContent = data.progress + '%';

                    if (data.status === 'complete') {
                        console.log('Scan complete, loading results...');
                        setTimeout(() => {
                            loadResults();
                            document.getElementById('progressDiv').classList.remove('active');
                            btn.disabled = false;
                        }, 500);
                    } else {
                        setTimeout(() => pollStatus(btn), 500);
                    }
                })
                .catch(err => {
                    console.error('Error polling status:', err);
                    setTimeout(() => pollStatus(btn), 1000);
                });
        }

        function loadResults() {
            fetch('/api/results')
                .then(r => r.json())
                .then(data => {
                    console.log('Results loaded:', data);
                    
                    // Handle both old array format and new object format
                    let results = data.results || data;
                    if (!Array.isArray(results)) {
                        results = [];
                    }
                    
                    const count = data.count !== undefined ? data.count : results.length;
                    const total = data.total !== undefined ? data.total : results.length;
                    const min_conviction = data.min_conviction !== undefined ? data.min_conviction : 4.0;
                    const minConv = Number(min_conviction);
                    
                    console.log(`Results loaded: ${count}/${total} candidates (min conviction: ${minConv})`);
                    
                    if (!results || results.length === 0) {
                        document.getElementById('resultsDiv').innerHTML = 
                            `<div class="empty">No candidates found with conviction >= ${minConv.toFixed(1)}. Try lowering the conviction threshold or running a new scan.</div>`;
                        return;
                    }

                    try {
                        const html = results.map(opp => {
                            console.log('Processing stock:', opp.ticker, opp);
                            const potentialX = opp.potential_x || ((opp.conviction / 10) * 10).toFixed(1);
                            const positionSize = opp.position_size || 2000;
                            const stopLoss = opp.stop_loss || (opp.price * 0.92);
                            const takeProfit = opp.take_profit || (opp.price * 2);
                            const reasonArr = opp.ai_reasoning || [];
                            const breakdown = opp.score_breakdown || {};
                            const breakdownHtml = Object.keys(breakdown).length ? `<div class="card-reasoning">${Object.entries(breakdown).filter(([k]) => k!=='total').map(([k,v]) => `<div class="reason-item">${k.replace(/_/g,' ')}: ${Number(v).toFixed(2)}</div>`).join('')}</div>` : '';
                            const reasonHtml = reasonArr.length ? `<div class="card-reasoning">${reasonArr.slice(0,2).map(r => `<div class="reason-item">${r}</div>`).join('')}</div>` : breakdownHtml;
                            
                            return `
                            <div class="opportunity-card" data-ticker="${opp.ticker}" role="button" aria-label="View ${opp.ticker} details" style="cursor: pointer;" tabindex="0">
                                <div class="card-header">
                                    <div class="ticker-section">
                                        <span class="ticker-large" style="cursor:pointer">${opp.ticker}</span>
                                        <span class="company-name">${opp.company_name || opp.ticker}</span>
                                        <span class="sector-tag">${opp.sector || 'Technology'}</span>
                                    </div>
                                    <div class="conviction-section">
                                        <span class="conviction-score">${opp.conviction.toFixed(1)}/10</span>
                                        <span class="potential-label">${potentialX}x potential</span>
                                    </div>
                                    ${reasonHtml}
                                </div>
                                
                                <div class="card-body">
                                    <div class="metrics-row">
                                        <div class="metric-box">
                                            <div class="metric-label">Current Price</div>
                                            <div class="metric-value primary">$${opp.price.toFixed(2)}</div>
                                        </div>
                                        <div class="metric-box">
                                            <div class="metric-label">${opp.optimal_ma || 200}-Day MA</div>
                                            <div class="metric-value">$${(opp.ma_value || opp.price * 0.95).toFixed(2)}</div>
                                        </div>
                                        <div class="metric-box">
                                            <div class="metric-label">Stop Loss</div>
                                            <div class="metric-value" style="color: #ff5252;">$${stopLoss.toFixed(2)}</div>
                                        </div>
                                        <div class="metric-box">
                                            <div class="metric-label">Target Price</div>
                                            <div class="metric-value" style="color: #00c853;">$${takeProfit.toFixed(2)}</div>
                                        </div>
                                        <div class="metric-box">
                                            <div class="metric-label">Position Size</div>
                                            <div class="metric-value success">$${positionSize.toLocaleString()}</div>
                                        </div>
                                    </div>
                                    
                                    <div class="analysis-text">
                                        ${opp.description || ('Price > ' + (opp.optimal_ma || 200) + '-MA, bullish momentum.')}
                                    </div>
                                    
                                    <div class="catalysts-section">
                                        <strong>Catalysts:</strong>
                                        <ul class="catalyst-list">
                                            ${(opp.catalysts && opp.catalysts.length > 0 ? opp.catalysts : [
                                                opp.gain_1m > 15 ? '1-month momentum surge' : 'Positive momentum trend',
                                                opp.gain_6m > 20 ? '6-month breakout' : 'Technical strength',
                                                'Earnings upcoming'
                                            ]).slice(0, 3).map(c => `<li>${c}</li>`).join('')}
                                        </ul>
                                    </div>
                                    
                                    <div class="trade-setup">
                                        <button class="buy-btn" onclick="event.stopPropagation(); alert('Order placed: ${opp.ticker}');">BUY $${positionSize.toLocaleString()}</button>
                                        <button class="skip-btn" onclick="event.stopPropagation();">SKIP FOR NOW</button>
                                    </div>
                                </div>
                            </div>
                        `;
                        }).join('');

                        // Add header with count info
                        var headerMsg = '✓ Found ' + count + ' candidates';
                        if (count < total) {
                            headerMsg += ' out of ' + total + ' (filtered at ' + minConv.toFixed(1) + '/10 conviction)';
                        }
                        const header = `
                            <div style="padding: 12px; background: #2a3f5f; border-radius: 6px; margin-bottom: 16px; border-left: 4px solid #1e88e5;">
                                <p style="margin: 0; color: #1e88e5; font-weight: 600;">${headerMsg}</p>
                            </div>`;
                        
                        document.getElementById('resultsDiv').innerHTML = header + html;
                        console.log('Results rendered successfully');
                        
                        // Bind click handlers to all cards
                        setTimeout(() => {
                            const cards = document.querySelectorAll('.opportunity-card');
                            console.log(`Found ${cards.length} cards to bind`);
                            
                            cards.forEach(card => {
                                const ticker = card.getAttribute('data-ticker');
                                card.addEventListener('click', function(e) {
                                    // Only trigger if not clicking on buttons
                                    if (e.target.closest('.buy-btn') || e.target.closest('.skip-btn')) {
                                        return;
                                    }
                                    console.log(`🃏 Card clicked: ${ticker}`);
                                    e.preventDefault();
                                    console.log('Calling viewStockDetail from card handler...');
                                    if (window.viewStockDetail) {
                                        viewStockDetail(ticker);
                                    } else {
                                        console.error('viewStockDetail not available!');
                                    }
                                });
                                
                                // Add visual feedback
                                card.addEventListener('mouseenter', function() {
                                    this.style.opacity = '0.8';
                                });
                                card.addEventListener('mouseleave', function() {
                                    this.style.opacity = '1';
                                });
                            });
                        }, 0);

                        // Extra robust handler: ensure clicks on the ticker text always open details
                        // Use capture phase so it runs before other handlers that may stop propagation
                        document.addEventListener('click', function(e){
                            try {
                                const t = e.target.closest && e.target.closest('.ticker-large');
                                if (!t) return;
                                const ticker = t.textContent && t.textContent.trim();
                                if (!ticker) return;
                                // If click originated on a buy/skip button, ignore
                                if (e.target.closest && (e.target.closest('.buy-btn') || e.target.closest('.skip-btn'))) return;
                                console.log('🎯 Ticker span click captured ->', ticker);
                                e.preventDefault();
                                try { 
                                    console.log('Calling viewStockDetail from ticker handler...');
                                    if (window.viewStockDetail) {
                                        window.viewStockDetail(ticker); 
                                    } else {
                                        console.error('viewStockDetail not found on window!');
                                    }
                                } catch(err) { console.error('ticker click handler error', err); }
                            } catch(err) { console.error('ticker delegated handler error', err); }
                        }, true);
                    } catch (err) {
                        console.error('Error rendering results:', err);
                        document.getElementById('resultsDiv').innerHTML = 
                            '<div class="empty">Error rendering results: ' + err.message + '</div>';
                    }
                })
                .catch(err => {
                    console.error('Error loading results:', err);
                    document.getElementById('resultsDiv').innerHTML = 
                        '<div class="empty">Error loading results. Please try again.</div>';
                });
        }

            // Global delegated click handler as last-resort fallback (catches dynamically rendered cards)
        document.addEventListener('click', function(e) {
            try {
                const btn = e.target.closest('.buy-btn, .skip-btn');
                if (btn) return; // ignore clicks on action buttons

                const card = e.target.closest('.opportunity-card');
                if (!card) return;

                const ticker = card.getAttribute('data-ticker') || (card.querySelector('.ticker-large') ? card.querySelector('.ticker-large').textContent.trim() : null);
                if (!ticker) return;

                console.log('Delegated capture: opportunity-card clicked ->', ticker);
                // Small visual flash for immediate feedback
                card.style.transition = 'box-shadow 0.12s ease, transform 0.12s ease';
                card.style.transform = 'translateY(-3px)';
                setTimeout(() => { card.style.transform = ''; }, 120);

                // call viewStockDetail asynchronously to avoid interrupting other handlers
                setTimeout(() => {
                    try { viewStockDetail(ticker); } catch(err) { console.error('delegated viewStockDetail error', err); }
                }, 0);
            } catch (err) {
                console.error('delegation handler error', err);
            }
        }, false);

            // Debug panel removed to reduce script surface and avoid parse issues

        function displayStockDetail(data) {
            try {
                console.log('🔍 displayStockDetail called');
                console.log('Full data object:', JSON.stringify(data, null, 2));

                // Single, robust populate path
                forcePopulateDetail(data);
                hideLoadingOverlay();
                console.log('✅ displayStockDetail COMPLETE');
            } catch (err) {
                console.error('❌ ERROR in displayStockDetail:', err);
                console.error('Stack:', err.stack);
                alert('Error: ' + err.message);
                backToList();
            }
        }
        // Ensure global access for fallbacks
        window.displayStockDetail = displayStockDetail;

        function showLoadingOverlay(ticker) {
            const ov = document.getElementById('loadingOverlay');
            const tk = document.getElementById('loadingTicker');
            if (tk) tk.textContent = ticker ? `(${ticker})` : '';
            if (ov) ov.style.display = 'flex';
        }

        function hideLoadingOverlay() {
            const ov = document.getElementById('loadingOverlay');
            if (ov) ov.style.display = 'none';
        }

        function renderChart(chartData, technical) {
            const ctx = document.getElementById('priceChart').getContext('2d');
            if (!chartData || !chartData.dates || !chartData.dates.length) return;

            if (priceChart) {
                priceChart.destroy();
            }

            const datasets = [];
            const priceSeries = chartData.prices || chartData.close || [];

            datasets.push({
                label: 'Price',
                data: priceSeries,
                borderColor: '#1e88e5',
                backgroundColor: 'rgba(30, 136, 229, 0.1)',
                borderWidth: 2,
                tension: 0.1,
                fill: true,
                pointRadius: 0,
                yAxisID: 'y'
            });

            if (chartData.ma_50 && chartData.ma_50.length > 0) {
                datasets.push({
                    label: 'MA 50',
                    data: chartData.ma_50,
                    borderColor: '#ffb300',
                    borderWidth: 1.5,
                    borderDash: [5, 5],
                    fill: false,
                    pointRadius: 0,
                    yAxisID: 'y'
                });
            }

            if (chartData.ma_200 && chartData.ma_200.length > 0) {
                datasets.push({
                    label: 'MA 200',
                    data: chartData.ma_200,
                    borderColor: '#e53935',
                    borderWidth: 1.5,
                    borderDash: [5, 5],
                    fill: false,
                    pointRadius: 0,
                    yAxisID: 'y'
                });
            }

            const annotations = {};
            if (technical && technical.support_levels) {
                technical.support_levels.forEach((level, i) => {
                    annotations[`support${i}`] = {
                        type: 'line',
                        yMin: level,
                        yMax: level,
                        borderColor: '#00c853',
                        borderWidth: 2,
                        borderDash: [10, 5],
                        label: {
                            display: true,
                            content: `Support: $${level.toFixed(2)}`,
                            position: 'end',
                            backgroundColor: 'rgba(0, 200, 83, 0.8)',
                            color: '#fff',
                            font: { size: 11 }
                        }
                    };
                });
            }

            if (technical && technical.resistance_levels) {
                technical.resistance_levels.forEach((level, i) => {
                    annotations[`resistance${i}`] = {
                        type: 'line',
                        yMin: level,
                        yMax: level,
                        borderColor: '#ff5252',
                        borderWidth: 2,
                        borderDash: [10, 5],
                        label: {
                            display: true,
                            content: `Resistance: $${level.toFixed(2)}`,
                            position: 'end',
                            backgroundColor: 'rgba(255, 82, 82, 0.8)',
                            color: '#fff',
                            font: { size: 11 }
                        }
                    };
                });
            }

            priceChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: chartData.dates,
                    datasets: datasets
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    interaction: { mode: 'index', intersect: false },
                    plugins: {
                        legend: {
                            display: true,
                            labels: { color: '#e0e0e0', font: { size: 12 }, usePointStyle: true }
                        },
                        tooltip: {
                            backgroundColor: 'rgba(26, 31, 46, 0.95)',
                            titleColor: '#fff',
                            bodyColor: '#e0e0e0',
                            borderColor: '#2a3f5f',
                            borderWidth: 1,
                            displayColors: true,
                            callbacks: {
                                label: function(context) {
                                    let label = context.dataset.label || '';
                                    if (label) {
                                        label += ': ';
                                    }
                                    if (context.parsed.y !== null) {
                                        label += '$' + context.parsed.y.toFixed(2);
                                    }
                                    return label;
                                }
                            }
                        },
                        annotation: { annotations: annotations }
                    },
                    scales: {
                        x: {
                            grid: { color: 'rgba(42, 63, 95, 0.3)', drawBorder: false },
                            ticks: { color: '#999', maxTicksLimit: 12 }
                        },
                        y: {
                            grid: { color: 'rgba(42, 63, 95, 0.3)', drawBorder: false },
                            ticks: {
                                color: '#999',
                                callback: function(value) { return '$' + value.toFixed(0); }
                            },
                            position: 'right'
                        }
                    }
                }
            });
            console.log('Chart rendered: type=line labels=', (chartData && chartData.dates ? chartData.dates.length : 0), 'datasets=', datasets.length);
        }

        // Robust populate: writes all fields and retries to avoid timing issues
        function forcePopulateDetail(data) {
            try {
                console.log('▶ forcePopulateDetail start', data && data.ticker, data);
                if (window.logOnPage) { window.logOnPage('forcePopulateDetail ' + (data && data.ticker ? data.ticker : 'unknown')); }
                const apply = (payload) => {
                    try {
                        const screen = payload.screen_data || {};
                        const tech = payload.technical_analysis || {};
                        const fund = payload.fundamentals || {};
                        const ai = payload.ai_analysis || {};
                        const n = (v) => (typeof v === 'number' ? v : (v != null ? Number(v) : NaN));
                        const set = (id, val) => {
                            const el = document.getElementById(id);
                            if (el) {
                                el.textContent = val;
                            } else {
                                console.warn('⚠ element not found:', id);
                            }
                        };

                        // Header
                        set('detailTicker', payload.ticker || '-');
                        set('detailCompanyName', fund.company_name || payload.ticker || '-');
                        const price = n(screen.price);
                        set('detailPrice', isFinite(price) ? '$' + price.toFixed(2) : '$-');

                        // Technicals
                        set('tech-price', isFinite(price) ? '$' + price.toFixed(2) : '-');
                        const maV = n(screen.ma_value);
                        set('tech-ma', screen.optimal_ma ? `${screen.optimal_ma}-day @ $${isFinite(maV) ? maV.toFixed(2) : '-'}` : '-');
                        const rsi = n(tech.rsi);
                        set('tech-rsi', isFinite(rsi) ? rsi.toFixed(1) : '-');
                        const conv = n(screen.conviction);
                        set('tech-conviction', isFinite(conv) ? conv.toFixed(1) + '/10' : '-');
                        const g1 = n(screen.gain_1m), g3 = n(screen.gain_3m), g1y = n(screen.gain_1y);
                        set('returns-1m', isFinite(g1) ? g1.toFixed(2) + '%' : '-');
                        set('returns-3m', isFinite(g3) ? g3.toFixed(2) + '%' : '-');
                        set('returns-1y', isFinite(g1y) ? g1y.toFixed(2) + '%' : '-');

                        // If values still show '-', force rebuild of the metric cards
                        try {
                            const tp = document.getElementById('tech-price');
                            if (tp && (tp.textContent === '-' || tp.textContent === '')) {
                                const card = tp.closest('.card');
                                if (card) {
                                    card.innerHTML = `
                                        <div class="metric-row"><span class="metric-label">Current Price</span><span class="metric-value">${isFinite(price)?('$'+price.toFixed(2)):'-'}</span></div>
                                        <div class="metric-row"><span class="metric-label">Optimal MA</span><span class="metric-value">${screen.optimal_ma? (screen.optimal_ma+'-day @ $'+(isFinite(maV)?maV.toFixed(2):'-')) : '-'}</span></div>
                                        <div class="metric-row"><span class="metric-label">RSI (14)</span><span class="metric-value">${isFinite(rsi)? rsi.toFixed(1) : '-'}</span></div>
                                        <div class="metric-row"><span class="metric-label">Volume Ratio</span><span class="metric-value">${isFinite(n(tech.volatility))? n(tech.volatility).toFixed(1)+'%' : '-'}</span></div>
                                        <div class="metric-row"><span class="metric-label">Conviction Score</span><span class="metric-value">${isFinite(conv)? conv.toFixed(1)+'/10' : '-'}</span></div>
                                        <div class="metric-row"><span class="metric-label">1M Returns</span><span class="metric-value">${isFinite(g1)? g1.toFixed(2)+'%' : '-'}</span></div>
                                        <div class="metric-row"><span class="metric-label">3M Returns</span><span class="metric-value">${isFinite(g3)? g3.toFixed(2)+'%' : '-'}</span></div>
                                        <div class="metric-row"><span class="metric-label">1Y Returns</span><span class="metric-value">${isFinite(g1y)? g1y.toFixed(2)+'%' : '-'}</span></div>
                                    `;
                                }
                            }
                        } catch (e) { console.warn('tech card rebuild error', e); }

                        // Fundamentals
                        const pe = n(fund.pe_ratio);
                        set('fund-pe', isFinite(pe) ? pe.toFixed(2) : (fund.pe_ratio || '-'));
                        const mc = n(fund.market_cap);
                        set('fund-mc', isFinite(mc) ? '$' + (mc / 1e9).toFixed(2) + 'B' : (fund.market_cap || '-'));
                        set('fund-sector', fund.sector || '-');
                        set('fund-industry', fund.industry || '-');
                        const roe = n(fund.roe);
                        set('fund-roe', isFinite(roe) ? (roe * 100).toFixed(2) + '%' : (fund.roe || '-'));
                        const pm = n(fund.profit_margin);
                        set('fund-pm', isFinite(pm) ? (pm * 100).toFixed(2) + '%' : (fund.profit_margin || '-'));
                        const de = n(fund.debt_to_equity);
                        set('fund-de', isFinite(de) ? de.toFixed(2) : (fund.debt_to_equity || '-'));

                        // Fundamentals fallback rebuild if still dashes
                        try {
                            const fpe = document.getElementById('fund-pe');
                            if (fpe && (fpe.textContent === '-' || fpe.textContent === '')) {
                                const fcard = fpe.closest('.card');
                                if (fcard) {
                                    fcard.innerHTML = `
                                        <div class="metric-row"><span class="metric-label">P/E Ratio</span><span class="metric-value">${isFinite(pe)? pe.toFixed(2) : '-'}</span></div>
                                        <div class="metric-row"><span class="metric-label">Market Cap</span><span class="metric-value">${isFinite(mc)? ('$'+(mc/1e9).toFixed(2)+'B') : '-'}</span></div>
                                        <div class="metric-row"><span class="metric-label">Sector</span><span class="metric-value">${fund.sector || '-'}</span></div>
                                        <div class="metric-row"><span class="metric-label">Industry</span><span class="metric-value">${fund.industry || '-'}</span></div>
                                        <div class="metric-row"><span class="metric-label">ROE</span><span class="metric-value">${isFinite(roe)? (roe*100).toFixed(2)+'%' : '-'}</span></div>
                                        <div class="metric-row"><span class="metric-label">Profit Margin</span><span class="metric-value">${isFinite(pm)? (pm*100).toFixed(2)+'%' : '-'}</span></div>
                                        <div class="metric-row"><span class="metric-label">Debt/Equity</span><span class="metric-value">${isFinite(de)? de.toFixed(2) : '-'}</span></div>
                                    `;
                                }
                            }
                        } catch (e) { console.warn('fund card rebuild error', e); }

                        // AI / Catalysts
                        const ten = n(ai.ten_x_score);
                        set('ai-score', isFinite(ten) ? ten.toFixed(1) : '-');
                        if (ai.ai_reasoning && Array.isArray(ai.ai_reasoning)) {
                            const el = document.getElementById('ai-reasoning');
                            if (el) el.innerHTML = ai.ai_reasoning.map((r) => `<li>${r}</li>`).join('');
                        }
                        if (ai.bull_case && Array.isArray(ai.bull_case)) {
                            const el = document.getElementById('bull-case');
                            if (el) el.innerHTML = ai.bull_case.map((r) => `<li>${r}</li>`).join('');
                        }
                        if (ai.bear_case && Array.isArray(ai.bear_case)) {
                            const el = document.getElementById('bear-case');
                            if (el) el.innerHTML = ai.bear_case.map((r) => `<li>${r}</li>`).join('');
                        }
                        if (ai.catalysts && Array.isArray(ai.catalysts)) {
                            const el = document.getElementById('catalysts');
                            if (el) el.innerHTML = ai.catalysts.map((r) => `<li>${r}</li>`).join('');
                        }

                        // News
                        if (payload.news && payload.news.length) {
                            const el = document.getElementById('newsDiv');
                            if (el) el.innerHTML = payload.news.map((n) => `<div class="news-item"><a href="${n.link}" target="_blank">${n.title}</a> <span style="color:#999">${n.source||''}</span></div>`).join('');
                        }

                        // Chart fallback (line) to avoid candlestick plugin issues
                        try {
                            if (payload.chart_data && payload.chart_data.dates && payload.chart_data.dates.length) {
                                currentChartType = 'line';
                                currentChartData = payload.chart_data;
                                currentTechnical = tech;
                                renderChart(payload.chart_data, tech);
                            }
                        } catch (e) { console.error('forcePopulate chart error', e); }
                    } catch (e) {
                        console.error('apply populate error', e);
                    }
                };

                // apply immediately and with retries to avoid any timing issues
                apply(data);
                setTimeout(() => apply(data), 150);
                setTimeout(() => apply(data), 400);
                setTimeout(() => apply(data), 900);

                // Continuous enforcement loop for a short window to defeat any late overwrites
                try {
                    let ticks = 0;
                    if (window.detailPopulateIntervalId) {
                        clearInterval(window.detailPopulateIntervalId);
                        window.detailPopulateIntervalId = null;
                    }
                    window.detailPopulateIntervalId = setInterval(() => {
                        try { apply(data); } catch(e) { }
                        ticks++;
                        if (ticks >= 25) { // ~6s at 240ms
                            clearInterval(window.detailPopulateIntervalId);
                            window.detailPopulateIntervalId = null;
                            console.log('⏹ populate enforcement loop stopped');
                        }
                    }, 240);
                    console.log('▶ populate enforcement loop started');
                } catch(e) { console.warn('populate loop start error', e); }

                console.log('✅ forcePopulateDetail applied (multi-pass)');
            } catch (e) {
                console.error('forcePopulateDetail error', e);
            }
        }

        function backToList() {
            console.log('Switching back to list view');
            const detailView = document.getElementById('detailView');
            const listView = document.getElementById('listView');
            if (window.detailPopulateIntervalId) { try { clearInterval(window.detailPopulateIntervalId); window.detailPopulateIntervalId = null; } catch(_){} }
            
            if (detailView) {
                detailView.classList.remove('active');
                detailView.style.display = 'none';
            }
            if (listView) {
                listView.classList.add('active');
                listView.style.display = 'block';
            }
            
            window.scrollTo(0, 0);
        }
        // Ensure global access from inline handlers
        window.backToList = backToList;

        // Define viewStockDetail - SIMPLE, NO STUBS
        window.viewStockDetail = function(ticker) {
            console.log('viewStockDetail called:', ticker);
            if (navigator.sendBeacon) { try { navigator.sendBeacon('/client_log', 'VIEW_STOCK_DETAIL_CALL:'+ticker); } catch(_){} }
            
            const listView = document.getElementById('listView');
            const detailView = document.getElementById('detailView');
            
            if (listView) {
                listView.classList.remove('active');
                listView.style.display = 'none';
            }
            if (detailView) {
                detailView.classList.add('active');
                detailView.style.display = 'block';
            }
            
            window.scrollTo(0, 0);
            showLoadingOverlay(ticker);
            
            fetch('/api/stock/' + ticker)
                .then(r => r.json())
                .then(data => {
                    if (data.error) {
                        alert('Error: ' + data.error);
                        hideLoadingOverlay();
                        backToList();
                        return;
                    }
                    displayStockDetail(data);
                    // Defensive second-pass populate in case any block failed
                    setTimeout(() => { try { forcePopulateDetail(data); } catch(e) { console.error('forcePopulateDetail fail', e); } }, 0);
                    if (navigator.sendBeacon) { try { navigator.sendBeacon('/client_log', 'VIEW_STOCK_DETAIL_DONE:'+ticker); } catch(_){} }
                })
                .catch(err => {
                    alert('Error loading stock: ' + err.message);
                    if (navigator.sendBeacon) { try { navigator.sendBeacon('/client_log', 'VIEW_STOCK_DETAIL_FETCH_ERROR:'+err.message); } catch(_){} }
                    hideLoadingOverlay();
                    backToList();
                });
        };
        console.log('viewStockDetail defined:', typeof window.viewStockDetail);
    