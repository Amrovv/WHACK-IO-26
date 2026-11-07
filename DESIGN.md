---
name: Climbing Coach
description: A live bouldering coach's logbook, read from across the room.
colors:
  graphite-page: "#16181b"
  pencil-ink: "#eef0ea"
  graphite: "#b4bac1"
  pencil: "#8c939a"
  faint-rule: "#33383e"
  strong-rule: "#565c63"
  field-book-yellow: "#f2c230"
  yellow-ink: "#2a2306"
  red-pen: "#ff5a6e"
  video-frame: "#0b0c0d"
  limb-left-hand: "#4fc3f7"
  limb-right-hand: "#b388ff"
  limb-left-foot: "#9ccc65"
  limb-right-foot: "#e0e0e0"
typography:
  display:
    fontFamily: "Atkinson Hyperlegible Next, system-ui, sans-serif"
    fontSize: "clamp(72px, min(5.6vw, 9vh), 96px)"
    fontWeight: 800
    lineHeight: 1.05
    letterSpacing: "-0.015em"
  figure:
    fontFamily: "Atkinson Hyperlegible Mono, ui-monospace, monospace"
    fontSize: "clamp(72px, min(5.6vw, 9vh), 96px)"
    fontWeight: 700
    lineHeight: 0.95
    fontFeature: "tnum"
  headline:
    fontFamily: "Atkinson Hyperlegible Next, system-ui, sans-serif"
    fontSize: "clamp(24px, min(2.6vw, 4.2vh), 44px)"
    fontWeight: 800
    letterSpacing: "-0.01em"
  title:
    fontFamily: "Atkinson Hyperlegible Next, system-ui, sans-serif"
    fontSize: "clamp(24px, min(2.6vw, 4.2vh), 44px)"
    fontWeight: 800
    lineHeight: 1.1
  body:
    fontFamily: "Atkinson Hyperlegible Next, system-ui, sans-serif"
    fontSize: "clamp(16px, min(1.45vw, 2.35vh), 24px)"
    fontWeight: 400
    lineHeight: 1.2
  label:
    fontFamily: "Atkinson Hyperlegible Next, system-ui, sans-serif"
    fontSize: "clamp(16px, min(1.45vw, 2.35vh), 24px)"
    fontWeight: 700
  meta:
    fontFamily: "Atkinson Hyperlegible Mono, ui-monospace, monospace"
    fontSize: "0.8em"
    fontWeight: 500
    fontFeature: "tnum"
rounded:
  none: "0px"
  dot: "50%"
spacing:
  gap: "clamp(12px, 1.4vw, 24px)"
  rule: "3px"
  sub-rule: "2px"
  entry-rule: "1px"
components:
  caption-band:
    backgroundColor: "{colors.field-book-yellow}"
    textColor: "{colors.yellow-ink}"
    typography: "{typography.display}"
    rounded: "{rounded.none}"
    height: "clamp(165px, 21vh, 220px)"
  caption-band-debrief:
    backgroundColor: "{colors.field-book-yellow}"
    textColor: "{colors.yellow-ink}"
    typography: "{typography.headline}"
    height: "clamp(165px, 34vh, 340px)"
  flag:
    textColor: "{colors.graphite}"
    typography: "{typography.label}"
  flag-boxed:
    textColor: "{colors.pencil-ink}"
    rounded: "{rounded.none}"
    padding: "0.1em 0.5em"
  live-hold-chip:
    backgroundColor: "{colors.pencil-ink}"
    textColor: "{colors.graphite-page}"
    rounded: "{rounded.none}"
    padding: "0 0.2em"
  rung-number:
    textColor: "{colors.pencil}"
    width: "1.65em"
  rung-number-now:
    backgroundColor: "{colors.pencil-ink}"
    textColor: "{colors.graphite-page}"
    width: "1.65em"
  latency-figure:
    textColor: "{colors.pencil-ink}"
    typography: "{typography.figure}"
  latency-figure-late:
    backgroundColor: "{colors.pencil-ink}"
    textColor: "{colors.graphite-page}"
    padding: "0 0.12em"
  attempt-top:
    textColor: "{colors.red-pen}"
    typography: "{typography.title}"
---

# Design System: Climbing Coach

## Overview

**Creative North Star: "The Tick List"**

The screen is a climber's logbook kept live by the coach, after dark. A graphite page carries light pencil ink; ruled lines, not cards, divide it. The coach's spoken line is the one field-book-yellow thing on screen, a full-width band at the bottom, and a red pen marks sends. Everything is sized to be read from 3 m away on a laptop or projector, using one three-step size ramp and nothing in between.

Density is low and deliberate: few elements, each large. Figures are set in a monospaced face with tabular numerals so latency, holds and times read like entries in a book. Motion is rationed so that one thing moves at a time: a new caption line slides in, and when an attempt ends a pencil tick is drawn onto its highpoint rung. Nothing else animates.

The system rejects the dark ops dashboard: rounded stat cards, neon accent glow, colour used as decoration. Fallback modes are labelled in plain words inside a ruled box, never hidden.

**Key Characteristics:**
- Graphite page, pencil ink, ruled lines instead of cards.
- One colour, one meaning: yellow is the coach's voice, red is a send, the problem's hold colour is the climber's live position.
- One fixed three-step size ramp (small, medium, large), tuned for 3 m.
- Words in Atkinson Hyperlegible Next, figures in Atkinson Hyperlegible Mono.
- Square corners; only dots are round.
- One motion at a time: the arriving caption and the drawn tick.

## Colors

A near-monochrome graphite logbook with two inks reserved for meaning, plus one colour borrowed from the wall.

### Primary
- **Field-Book Yellow** (field-book-yellow): the coach's voice and nothing else. It fills the caption band and is the text-selection colour. Text on it is always Yellow Ink.
- **Yellow Ink** (yellow-ink): dark olive-brown ink used only on yellow: the caption text, the boxed "Coach" speaker tag, the latency in the band. Idle and waiting captions fade it to 62–70% alpha rather than switching colour.

### Secondary
- **Red Pen** (red-pen): marks a send (top) only: the red tick on the ladder, the "Top" outcome in the attempts list, the topped line in the timeline. Never used for errors or warnings. One exception, at the user's request: the "Live · camera N" mode flag (live webcam demo only) is red with a slowly flashing dot (steady under reduced motion).

**Mode flag.** The running head says what is running, from the server's hello: "Live · camera N" (red, flashing) for a webcam; "Analysing in real time" (ink, steady dot) for a pre-recorded clip analysed live, with a clip-progress strip (name, time, loop) along the bottom of the video; "Video test" (boxed) for `web/stream_test.py`, where the layout drops to the video alone with stream stats; "Session replay · <log>" (boxed) for a bus-log replay. Never call a clip "Recorded": the analysis is real time. The connection flag appears only when something is wrong ("Reconnecting…").

### Tertiary
- **Live Hold** (runtime, defaults to Pencil Ink): the problem's own hold colour, set from the hold map, fills the live rung number and the hold chip in the running head. Problem colours that would read as Red Pen or Field-Book Yellow (pink, red, orange, yellow) fall back to the Pencil Ink block. Its ink flips to Graphite Page or white by luminance.
- **Limb colours** (limb-left-hand, limb-right-hand, limb-left-foot, limb-right-foot): sky blue, lilac, leaf green and pale grey. Used only on the video overlay to tie a limb dot to the hold it is on. They never appear in the page chrome.

### Neutral
- **Graphite Page** (graphite-page): the page background, and the ink on any inverted light block.
- **Pencil Ink** (pencil-ink): primary text, emphasised figures, and the light block for inversions (live hold, late latency).
- **Graphite** (graphite): secondary text: section headings, flags, attempt details, the fall slash, latency bars.
- **Pencil** (pencil): tertiary text: unlit rung numbers, separators, timeline meta, suppressed entries, empty states.
- **Strong Rule** (strong-rule): the 3px structural rules under the running head and left of the side column.
- **Faint Rule** (faint-rule): the 2px and 1px rules between rungs, sections and entries, and the video frame outline.
- **Video Frame** (video-frame): the near-black well behind the camera picture.

### Named Rules
**The One Colour, One Meaning Rule.** Yellow is the coach speaking, red is a send, the hold colour is where the climber is now. No colour is used for a second purpose, and no decorative colour is added.

**The Inversion Rule.** Emphasis and alarm are shown by inverting to a light Pencil Ink block with Graphite Page ink (the live hold, a late latency figure), not by adding a warning colour.

## Typography

**Display Font:** Atkinson Hyperlegible Next (with system-ui, sans-serif), self-hosted, weights 400–800
**Label/Mono Font:** Atkinson Hyperlegible Mono (with ui-monospace, monospace), self-hosted, weights 500–800

**Character:** A legibility-first sans for words and its monospaced sibling for every figure, so numbers line up and read at distance. Weight does the hierarchy work: 800 for what must be read first, 700 for figures and headings, 400 for running text.

### Hierarchy
- **Display** (800, large step, 1.05, -0.015em): the coach's line in the caption band, up to two lines, balanced wrap.
- **Figure** (Mono 700, large step, 0.95, tabular): the latest cue latency in the side column.
- **Headline** (800, medium step, -0.01em): the running head ("Attempt 3 · hold 6 of 10 · 0:14"); also the debrief caption at 600 / 1.2, up to five lines, stepping down to 80% for long debriefs.
- **Title** (800, medium step, 1.1): attempt outcome words ("Climbing", "Top", "Fall"); rung numbers on the ladder (Mono 700).
- **Body** (400, small step, 1.2): timeline entries, attempt details, latency summary.
- **Label** (700, small step): section headings ("Cue latency", "Attempts", "Timeline"), sentence case. Flags use 600.
- **Meta** (Mono, 0.8em of body): timeline right-hand meta (ms, suppression reason).

### Named Rules
**The Three Steps Rule.** Every size is one of three steps: small (clamp 16–24px), medium (clamp 24–44px), large (clamp 72–96px). Below 900px wide they fix to 16 / 24 / 36px. Do not add a fourth step.

**The Mono Figures Rule.** Any number a viewer reads as data (ms, hold ids, times, counts, attempt numbers) is set in Atkinson Hyperlegible Mono with tabular numerals.

## Layout

A full-viewport, non-scrolling grid on wide screens: a running head across the top, then three columns (video stage taking the remaining width, the hold ladder at clamp 130–200px, the ruled side column at clamp 260–420px), then the full-width caption band at the bottom (clamp 165–220px tall, growing to clamp 165–340px during a debrief). The head is clamp 56–88px tall.

The video frame is sized by script to the stream's aspect ratio. The hold ladder is its own column, separate from the picture: a 3px Strong Rule on its left, the section padding, and a "Holds" title with the hold count in Pencil, like the side column's sections. Rungs share the column's height equally, top hold at the top.

The side column stacks three ruled sections (latency, attempts, timeline); the timeline takes the remaining height, scrolls, and fades its last 1.6em when it overflows.

A source bar sits in its own row under the picture: "Source", the clip picker (square, 2px Strong Rule, Pencil Ink on the raised page) and the clip time in Mono. It names what is being analysed and never sits on top of the picture; only the 5px progress bar does.

**Portrait video** (a picture taller than it is wide, on a wide screen) gets its own grid. The picture runs the full height on the left, and its column is only as wide as the picture. To its right, top to bottom: the source bar across the full width; then the hold ladder column; then the side data split in two (latency above attempts, and the timeline beside them, divided by a 2px Faint Rule); then the caption band, which spans only this right-hand side. 3px Strong Rules divide the picture's column from the rest.

Spacing comes from one fluid gap (clamp 12–24px), used at multiples of 0.6, 0.65, 1.2, 1.4 and 1.8 for section and band padding.

Below 900px the page stacks and scrolls in this order: head, caption, video (full width, at most 70% of the screen height), source bar, ladder, side column. The ladder turns horizontal, running from hold 1 on the left to the top hold on the right, and the ladder's and side column's left rules become top rules.

### Named Rules
**The Ruled, Not Carded Rule.** Regions are separated by rules: 3px Strong Rule for structure, 2px Faint Rule between rungs and sections, 1px Faint Rule between entries. Never wrap content in a filled or bordered card.

## Elevation & Depth

The system is flat. There are no drop shadows, glows or layered surfaces; the page is one graphite sheet and depth is not used. The only `box-shadow` in the build is an inset 3px Pencil Ink ring that outlines the highpoint rung number, which is a drawn box, not elevation.

### Named Rules
**The Flat Page Rule.** Nothing floats. Hierarchy comes from size, weight, rules and inversion, never from shadow.

## Shapes

Square corners everywhere (0px): the caption band, boxed flags, the speaker tag, the live-hold chip, rung numbers, the video frame. The only round forms are dots: the connection status dot and the overlay's limb dots and keypoints (50%), plus the round line caps of the hand-drawn marks. Boxes are drawn with a border of the text's own ink (2px on flags, 3px on the speaker tag and highpoint ring), never filled with a tint.

Pencil marks are stroked SVG paths, not glyphs: a short slash for a fall or down-climb, a check tick for a send, each followed by the attempt number in Mono.

## Components

### Caption Band
The coach's voice: a full-width Field-Book Yellow band at the bottom of the grid.
- **Layout:** speaker column on the left (the "Coach" tag above the mood word), the line in the middle, the latency in ms on the right in Mono at the medium step.
- **States:** cue (display type, two lines max; a cue that won't fit steps down once to 78% of the large step, three lines below 900px); debrief (taller band, medium step at 600, five lines max, seven when long); waiting/idle (ink at 62–70% alpha).
- **Motion:** each new line arrives once, 0.45s with a 0.18em rise from 35% opacity on `cubic-bezier(0.16, 1, 0.3, 1)`. Disabled under reduced motion.
- **Speaker tag:** 800 weight, uppercase, 0.1em tracking, boxed in a 3px Yellow Ink border. It names who is speaking; it is the one uppercase label in the chrome and is not a pattern for section headings.

### Running Head
Guide words for the climb, like the top of a book page.
- **Style:** headline type in Pencil Ink, parts separated by a Pencil-coloured middle dot at 400 weight. Single line, ellipsised.
- **Live hold chip:** the hold number in an inverted block filled with the live-hold colour.
- **Flags (right):** small-step 600 in Graphite. Fallback modes ("Colour-only holds", "Offline: no LLM") are boxed in a 2px Pencil Ink border with Pencil Ink text. Connection shows a hollow dot that fills when live.

### Hold Ladder (signature)
The problem as a column of rungs, top hold at the top, each attempt ticked at its highpoint.
- **Rung:** separated by 2px Faint Rules; the hold number in Mono 700, Pencil, in a 1.65em cell.
- **Highpoint:** number in Pencil Ink inside an inset 3px ring.
- **Now:** number inverted on the live-hold colour.
- **Marks:** Graphite slash for a fall, Red Pen tick for a send (stroke 3.2 and 3.6), with the attempt number. A newly finished attempt's mark draws in over 0.7s after a 0.15s delay.

### Latency Readout
- **Figure:** latest cue latency at the large step in Mono, with "ms" at the small step in Graphite.
- **Late:** over the 800ms target the figure inverts to a Pencil Ink block with Graphite Page ink, and its bar in the history fills Pencil Ink instead of Graphite.
- **History:** a row of thin bars with a dashed 1.5px Pencil Ink target line; a median and over-target count sit below in body type.

### Attempts List
- **Open entry:** the attempt in progress shows its number and "Climbing" in Pencil Ink with this attempt's event and spoken counts.
- **Finished entries:** whole rows, never cropped, carrying the same pencil mark as the ladder, the outcome word at the medium step (Red Pen for a top), and hold, duration and stalls below in Graphite. Entries are divided by 1px Faint Rules.

### Timeline
One line per entry, divided by 1px Faint Rules, meta in Mono on the right.
- **Spoken:** 700 weight. **Attempt boundaries:** 700 Pencil Ink, Red Pen for a top. **Suppressed:** Pencil with a 1.5px strike-through, hideable by the toggle. **Debrief:** italic Graphite, one ellipsised line.
- **Toggle:** a native checkbox tinted Pencil Ink, with a 3px Pencil Ink focus outline at 2px offset.

### Video Overlay
Drawn on a canvas over the camera picture, scaled with the picture.
- **Holds:** white outlined rectangles (80% alpha), numbered. Start and top holds get a white label with dark ink ("1 START", "10 TOP"); mid holds get a dark translucent label with white ink. A held hold takes its limb's colour as outline and a translucent fill.
- **Pose:** white skeleton (88% alpha, 4px round caps), a ring at the hip, limb dots in the limb colours with a dashed line to the hold they are on.

## Do's and Don'ts

### Do:
- **Do** reserve Field-Book Yellow for the coach's spoken or debrief line and Red Pen for sends.
- **Do** use only the three size steps (small, medium, large) and their fixed values below 900px.
- **Do** set every data figure in Atkinson Hyperlegible Mono with tabular numerals.
- **Do** separate regions with 3px, 2px and 1px rules, and keep corners square.
- **Do** show emphasis and lateness by inverting to a Pencil Ink block.
- **Do** label fallback modes in plain words in a 2px boxed flag.
- **Do** keep motion to the arriving caption and the drawn tick, and honour reduced motion.

### Don't:
- **Don't** build rounded stat cards, filled panels or a neon accent; this is the dashboard the system refuses.
- **Don't** add drop shadows or glows.
- **Don't** use yellow or red for warnings, errors or decoration.
- **Don't** let the problem's hold colour appear anywhere except the climber's live position.
- **Don't** use the limb colours outside the video overlay.
- **Don't** animate anything other than the caption arrival and the pencil tick.

### Known gaps (shipped, not rules)
- The last timeline row can be sliced by the caption band, and suppressed entries can wrap to two lines.
- Overlay hold labels can collide with limb dots, other labels or the "No video" note, especially at phone width.
