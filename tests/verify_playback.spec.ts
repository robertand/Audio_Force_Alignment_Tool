import { test, expect } from '@playwright/test';
import path from 'path';
import fs from 'fs';

test('Verify Timeline View Toggle and Blocks', async ({ page }) => {
    await page.goto('http://localhost:5000');
    const metadataPath = path.resolve('tests/test_data.csv');
    fs.writeFileSync(metadataPath, 'start,end,speaker,en,ro\n0.0,2.0,SpeakerA,Hello,Salut\n5.0,7.0,SpeakerA,Bye,Pa');
    await page.setInputFiles('#metadataFile', metadataPath);
    await page.click('#analyzeBtn');
    await expect(page.locator('#step2')).toBeVisible();
    const audioPath = path.resolve('tests/test_audio.wav');
    await page.setInputFiles('.audio-input[data-speaker="SpeakerA"]', audioPath);
    await page.check('input[value="none"]');
    await page.click('#processBtn');
    await expect(page.locator('#step3')).toBeVisible({ timeout: 30000 });

    // Screenshot 1: Detailed View
    await page.screenshot({ path: 'tests/detailed_view.png' });

    // Switch to Timeline View
    await page.click('#viewTimelineBtn');
    await expect(page.locator('#timelineView')).toBeVisible();

    // Wait for WaveSurfer to potentially render
    await page.waitForTimeout(2000);

    // Screenshot 2: Timeline View
    await page.screenshot({ path: 'tests/timeline_view.png' });
});
