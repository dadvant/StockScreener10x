from playwright.sync_api import sync_playwright

URL = "http://localhost:5000/"

def text(el):
    return el.inner_text().strip() if el else ""

def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        # Capture console logs for debugging
        page.on("console", lambda msg: print(f"[console] {msg.type}: {msg.text}"))
        page.on("pageerror", lambda exc: print(f"[pageerror] {getattr(exc, 'name', 'Error')}: {getattr(exc, 'message', str(exc))}\n{getattr(exc, 'stack', '')}"))
        page.goto(URL, wait_until="domcontentloaded")
        page.wait_for_timeout(800)
        # Check for global script error banner and log it
        try:
            banners = page.locator("text=Script error:")
            count = banners.count()
            if count and count > 0:
                for i in range(count):
                    try:
                        print("[banner]", banners.nth(i).inner_text())
                    except Exception:
                        pass
        except Exception:
            pass
        # Wait for main script to define viewStockDetail, then open AAPL
        page.wait_for_function("() => !!window.viewStockDetail", timeout=8000)
        page.evaluate("() => window.viewStockDetail('AAPL')")
        page.wait_for_selector("#detailView", state="visible", timeout=8000)
        # Wait until the ticker field is populated (not '-')
        page.wait_for_function("() => { const el = document.getElementById('detailTicker'); return el && el.textContent && el.textContent.trim() !== '-'; }", timeout=8000)

        # Grab core fields
        ticker = text(page.query_selector("#detailTicker"))
        company = text(page.query_selector("#detailCompanyName"))
        price = text(page.query_selector("#detailPrice"))
        tech_price = text(page.query_selector("#tech-price"))
        tech_ma = text(page.query_selector("#tech-ma"))
        rsi = text(page.query_selector("#tech-rsi"))
        conviction = text(page.query_selector("#tech-conviction"))
        pe = text(page.query_selector("#fund-pe"))
        mc = text(page.query_selector("#fund-mc"))
        sector = text(page.query_selector("#fund-sector"))
        industry = text(page.query_selector("#fund-industry"))

        assert ticker and ticker != "-", f"Ticker not populated (got '{ticker}')"
        assert company and company != "-", "Company not populated"
        assert price and price.startswith("$"), "Price not populated"
        assert tech_price and tech_price.startswith("$"), "Tech price not populated"
        assert tech_ma and ("-day" in tech_ma), "MA not populated"
        assert rsi and rsi != "-", "RSI not populated"
        assert conviction and "/10" in conviction, "Conviction not populated"
        assert pe and pe != "-", "PE not populated"
        assert mc and (mc.startswith("$") or mc != "-"), "Market cap not populated"
        assert sector and sector != "-", "Sector not populated"
        assert industry and industry != "-", "Industry not populated"

        # Chart canvas exists
        assert page.query_selector("canvas#priceChart") is not None, "Chart canvas missing"

        print("UI validation passed:")
        print({
            "ticker": ticker,
            "company": company,
            "price": price,
            "tech_price": tech_price,
            "tech_ma": tech_ma,
            "rsi": rsi,
            "conviction": conviction,
            "pe": pe,
            "mc": mc,
            "sector": sector,
            "industry": industry
        })

        browser.close()

if __name__ == "__main__":
    main()
