---
version: 1
slug: "web-index-html"
primary_target: "web/index.html"
related_targets: []
---

# Surface: live dashboard (web/index.html)

Mode: Operate (a working screen, but watched passively). Primary viewer: judges at the demo, on a laptop screen read from 3 m or more. Lead: the climber video plus the coach's spoken line, huge. Must stay visible: cue latency, attempt history, event timeline, current attempt counts. Keep every data topic, `?demo=1`, `?offline=1`, `?ws=`, `?video=` and the hide-suppressed toggle.

Memorable moment: the hold ladder beside the video, where each attempt leaves a tick at its highpoint and the live hold is lit in the problem's colour.

## Direction contract

THESIS: The coach keeps the climber's logbook live. Refuses the dark ops dashboard of rounded stat cards with a neon accent.

OWN-WORLD: Dark theme (user's call, 2026-11-07): graphite page (#16181B) with light pencil ink (#EEF0EA) and 3px mid-grey rules (#565C63) instead of cards. Field-book yellow (#F2C230) is the coach's voice only: the caption band, dark ink on yellow. Red pen (#FF5A6E) marks sends (tops). The problem's own hold colour marks only the climber's live position. Late latency inverts to a light block. Atkinson Hyperlegible Next for words, Atkinson Hyperlegible Mono for figures. Square corners. One fixed three-step size ramp, sized for 3 m (raise from the star atlas).

STORY: A judge sees the climber, hears a cue and sees the same line arrive big, with its latency in milliseconds. They see how high this attempt is against earlier ones, and they trust it because fallback modes are labelled in plain words.

FIRST VIEWPORT: A running head across the top names the place, like a book's guide words (raise from the lexicon): "Attempt 3 · hold 6 of 10 · 0:14", with mode labels on the right. The video takes about two thirds of the width. The hold ladder, top hold at the top, sits right of the video. A ruled side column holds latency (72px+ figure with a target-lined history), the attempts list (the attempt in progress is an open entry carrying this attempt's event and spoken counts; finished attempts are whole rows with the same pencil marks as the ladder, never cropped: raise from the doujin catalog) and a one-line-per-entry timeline. A full-width yellow caption band at the bottom carries the line at 72px+ with its ms. No primary action: the viewer only watches.

FORM: Tick List (climber's logbook), candidate 4 of 7 on my list; seed: 5322f0f36e27a42e. Raises: one colour, one meaning (warm consumer app); one thing moves at a time, so a new cue is the only motion (datamatics).
Signature interaction: when an attempt ends, a pencil tick is drawn onto its highpoint rung, and the caption band slides in once per new line.

FINISH: unreviewed and undocumented is unfinished; this build ends with the finish review, the verdict, DESIGN.md, and every shipping raster carrying its provenance
