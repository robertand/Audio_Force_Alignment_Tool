const { chromium } = require('playwright');
const fs = require('fs');
const path = require('path');

(async () => {
    const browser = await chromium.launch();
    const page = await browser.newPage();

    // Set up a mock API response for status polling and waveform peaks
    await page.route('**/api/job/*', route => {
        route.fulfill({
            status: 200,
            contentType: 'application/json',
            body: JSON.stringify({
                status: '✅ Completed',
                progress: 100,
                results: [
                    {
                        speaker: 'SPEAKER_01',
                        text_ro: 'Bună ziua, acesta este un test.',
                        text_en: 'Hello, this is a test.',
                        csv_start: 10.0,
                        csv_end: 15.0,
                        source_start: 2.5,
                        source_end: 7.5,
                        confidence: 0.95
                    },
                    {
                        speaker: 'SPEAKER_02',
                        text_ro: 'Salut! Funcționează corect.',
                        text_en: 'Hi! It works correctly.',
                        csv_start: 16.0,
                        csv_end: 20.0,
                        source_start: 1.0,
                        source_end: 5.0,
                        confidence: 0.88
                    }
                ]
            })
        });
    });

    // Mock waveform data
    await page.route('**/api/waveform-data/*', route => {
        // Generate dummy peaks (min/max pairs between -1 and 1)
        const peaks = [];
        for (let i = 0; i < 200; i++) {
            peaks.push(-Math.random(), Math.random());
        }
        route.fulfill({
            status: 200,
            contentType: 'application/json',
            body: JSON.stringify({
                peaks: peaks,
                duration: 60.0
            })
        });
    });

    // Go to the page
    await page.goto('http://localhost:5000');

    page.on('console', msg => console.log('PAGE LOG:', msg.text()));
    page.on('pageerror', err => console.log('PAGE ERROR:', err.message));

    // Check if WaveSurfer is loaded
    const wsLoaded = await page.evaluate(() => typeof WaveSurfer !== 'undefined');
    console.log('WaveSurfer defined in page:', wsLoaded);

    // Manually trigger step 3 by setting global state and calling showStep3
    // We can do this by executing script in the browser context
    await page.evaluate(() => {
        currentJobId = 'test-job-id';
        // Mock data usually comes from backend
        const mockResults = [
            {
                speaker: 'SPEAKER_01',
                text_ro: 'Bună ziua, acesta este un test.',
                text_en: 'Hello, this is a test.',
                csv_start: 10.0,
                csv_end: 15.0,
                source_start: 2.5,
                source_end: 7.5,
                confidence: 0.95
            },
            {
                speaker: 'SPEAKER_02',
                text_ro: 'Salut! Funcționează corect.',
                text_en: 'Hi! It works correctly.',
                csv_start: 16.0,
                csv_end: 20.0,
                source_start: 1.0,
                source_end: 5.0,
                confidence: 0.88
            }
        ];
        jobAlignmentResults = mockResults;
        showStep3(mockResults);
    });

    // Wait for segments to render
    await page.waitForSelector('#seg-0');

    // Switch to Timeline View
    await page.click('#viewTimelineBtn');

    // Wait for timeline to render
    await page.waitForSelector('#timeline-ws-project', { state: 'visible' });
    // Give WaveSurfer time to initialize plugins and render regions
    await page.waitForTimeout(2000);

    const containerHtml = await page.$eval('#timeline-ws-project', el => el.innerHTML);
    console.log('Container HTML structure length:', containerHtml.length);
    if (containerHtml.length < 500) {
        console.log('Container HTML snippet:', containerHtml);
    }

    // Log the number of regions found in the DOM (including shadow DOM if applicable)
    const regionCount = await page.evaluate(() => {
        const results = [];
        // Shadow DOM piercing selector
        const regions = document.querySelector('#timeline-ws-project')?.shadowRoot?.querySelectorAll('[part="region"]');
        if (regions) results.push(...regions);

        // Also check if WaveSurfer 7 uses a nested shadow root or specific elements
        const wsWrapper = document.querySelector('#timeline-ws-project')?.querySelector('div')?.shadowRoot;
        if (wsWrapper) {
            const shadowRegions = wsWrapper.querySelectorAll('[part="region"]');
            results.push(...shadowRegions);
        }

        return results.length;
    });
    console.log(`Regions found in DOM: ${regionCount}`);

    // Give canvas some time to draw mini waveforms
    await page.waitForTimeout(1000);

    const screenshotDir = '/home/jules/verification';
    if (!fs.existsSync(screenshotDir)) {
        fs.mkdirSync(screenshotDir, { recursive: true });
    }

    await page.screenshot({ path: path.join(screenshotDir, 'project_timeline_v2_final.png'), fullPage: true });

    // Drag a region on the timeline (simulate user interaction)
    // Use part=region for Shadow DOM piercing in Playwright
    const region = page.locator('[part="region"]').first();
    if (await region.count() > 0) {
        const box = await region.boundingBox();
        if (box) {
            await page.mouse.move(box.x + 10, box.y + 10);
            await page.mouse.down();
            await page.mouse.move(box.x + 50, box.y + 10);
            await page.mouse.up();
            await page.waitForTimeout(500);
        }
    }

    // Switch back to Detailed View
    await page.click('#viewDetailedBtn');

    // Verify sync back (check if csv-start value changed from 10.0)
    const csvStartVal = await page.$eval('#seg-0 .csv-start', el => el.value);
    console.log('Synced CSV Start:', csvStartVal);

    await page.screenshot({ path: path.join(screenshotDir, 'detailed_view_after_sync.png'), fullPage: true });

    await browser.close();
})();
