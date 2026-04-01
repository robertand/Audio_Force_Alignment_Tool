import asyncio
from playwright.async_api import async_playwright
import os

async def run_verification():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()

        # 1. Load app
        await page.goto("http://localhost:5000")
        print("Page loaded")

        # 2. Upload metadata
        metadata_path = os.path.abspath("verification_metadata.csv")
        await page.set_input_files("#metadataFile", metadata_path)
        await page.click("#analyzeBtn")
        print("Metadata analyzed")

        # 3. Step 2: Upload Audio
        await page.wait_for_selector("#step2", state="visible")
        audio_path = os.path.abspath("verification_audio.wav")
        # Find SpeakerA input
        await page.set_input_files('input.audio-input[data-speaker="SpeakerA"]', audio_path)
        await page.click('input[value="none"]')
        await page.click("#processBtn")
        print("Processing started")

        # 4. Step 3: Editor
        await page.wait_for_selector("#step3", state="visible", timeout=60000)
        print("Step 3 visible")

        # Switch to timeline
        await page.click("#viewTimelineBtn")
        print("Switched to timeline")

        # Wait for regions to be rendered in JS
        await page.wait_for_function("() => window.projectRegions && window.projectRegions.getRegions().length > 0")
        print("Regions detected in JS")

        # Give it a second to render
        await asyncio.sleep(2)

        # Take screenshot of the timeline container
        timeline_handle = await page.query_selector("#projectTimeline")
        await timeline_handle.screenshot(path="verification/timeline_view.png")
        print("Timeline screenshot saved")

        # Verify Bidirectional Sync (JS Side)
        # Move region 0 in JS
        await page.evaluate("""() => {
            const region = window.projectRegions.getRegions().find(r => r.id === 'csv-region-0');
            region.setOptions({ start: 2, end: 5 });
            window.projectRegions.emit('region-updated', region);
        }""")
        print("Region 0 moved via JS")

        await asyncio.sleep(1)

        # Check if detailed view input updated
        csv_start_val = await page.locator("#seg-0 .csv-start").input_value()
        print(f"Detailed view CSV Start for seg-0 after sync: {csv_start_val}")

        # Test Playback Highlight
        await page.click("#playBtn")
        print("Playback started")
        # Region 0 is at 2-5s now. Let's wait until 3.5s.
        await asyncio.sleep(3.5)

        is_highlighted = await page.evaluate("""() => {
            return window.activeProjectAudio.has(0);
        }""")
        print(f"Is region 0 highlighted at ~3.5s? {is_highlighted}")

        await timeline_handle.screenshot(path="verification/timeline_playback.png")

        await browser.close()

if __name__ == "__main__":
    if not os.path.exists("verification"):
        os.makedirs("verification")
    asyncio.run(run_verification())
