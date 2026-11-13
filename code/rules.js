// rules.js — SWRM rule storage and editing (v8ui object)
//
// Holds one base rule and one target rule per cohort. A rule is 80 floats:
//   [ freq0.xyzw, amp0.xyzw, freq1.xyzw, amp1.xyzw, ..., freq9.xyzw, amp9.xyzw ]
// All rules live in a single Float32Array laid out exactly like the
// cohort_rules texture (4 planes, 20 columns, 65 rows):
//   slot 0     = base rule      (row 0)
//   slot c + 1 = target of cohort c (row c + 1)
//
// The v8ui box shows every rule as a heatmap (one row per slot, one column per
// float) and the numeric detail of the selected row. Click a row to select it.

autowatch = 1;
inlets = 1;
outlets = 1;

mgraphics.init();
mgraphics.relative_coords = 0;
mgraphics.autofill = 0;

//------------------------------------ LAYOUT ----------------------------------

const N_CENTERS = 10;
const FLOATS_PER_VEC4 = 4;
const FLOATS_PER_CENTER = 2 * FLOATS_PER_VEC4;          // freq + amp
const FLOATS_PER_RULE = N_CENTERS * FLOATS_PER_CENTER;  // 80
const MAX_COHORTS = 64;
const N_SLOTS = MAX_COHORTS + 1;                        // base + targets
const BASE_SLOT = 0;

const rules = new Float32Array(N_SLOTS * FLOATS_PER_RULE);

let n_cohorts_value = 1;

function cohortSlot(cohort) {
    return cohort + 1;
}

function ruleOffset(slot) {
    return slot * FLOATS_PER_RULE;
}

// Offset of center i's frequency (amplitude is at + FLOATS_PER_VEC4)
function centerOffset(slot, i) {
    return ruleOffset(slot) + i * FLOATS_PER_CENTER;
}

// View on one rule (shares memory with `rules`)
function ruleView(slot) {
    return rules.subarray(ruleOffset(slot), ruleOffset(slot) + FLOATS_PER_RULE);
}

function clampInt(v, lo, hi) {
    return Math.max(lo, Math.min(hi, Math.floor(v)));
}

//------------------------------------ PRNG ------------------------------------

// 32-bit integer mix (lowbias32): turns any (seed, stream) pair into a
// well-spread PRNG state, so nearby seeds give unrelated rules.
function hash32(x) {
    x = x >>> 0;
    x ^= x >>> 16;
    x = Math.imul(x, 0x7feb352d);
    x ^= x >>> 15;
    x = Math.imul(x, 0x846ca68b);
    x ^= x >>> 16;
    return x >>> 0;
}

// Seeded PRNG (mulberry32), returns floats in [0, 1).
// `stream` separates independent sequences from the same seed (e.g. per cohort).
function makeRng(seed, stream) {
    let s = hash32(hash32(Math.floor(seed)) ^ hash32((stream | 0) + 0x9e3779b9));
    return function () {
        s = (s + 0x6d2b79f5) >>> 0;
        let t = s;
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}

const STREAM_BASE = 0;
function streamCohort(cohort) {
    return cohort + 1;
}

//------------------------------------ GENERATION ------------------------------

// Fluoddity's generate_random_centers: frequencies biased toward low values
// (scale 1..3, same draw as freq.x, as in the original), amplitudes in [-1, 1].
function randomRule(slot, rng) {
    for (let i = 0; i < N_CENTERS; i++) {
        const o = centerOffset(slot, i);
        const h0 = rng();
        const freq_scale = 1.0 + 2.0 * h0 * h0;
        rules[o + 0] = (h0 * 2.0 - 1.0) * freq_scale;
        rules[o + 1] = (rng() * 2.0 - 1.0) * freq_scale;
        rules[o + 2] = (rng() * 2.0 - 1.0) * freq_scale;
        rules[o + 3] = (rng() * 2.0 - 1.0) * freq_scale;
        for (let k = 0; k < 4; k++) {
            rules[o + 4 + k] = rng() * 2.0 - 1.0;
        }
    }
}

// Fluoddity's mutate_rule: dst = src mutated by `amount`.
// Additive noise on amplitudes, one multiplicative factor per center on frequencies.
function mutatedRule(dstSlot, srcSlot, amount, rng) {
    for (let i = 0; i < N_CENTERS; i++) {
        const d = centerOffset(dstSlot, i);
        const s = centerOffset(srcSlot, i);
        const freq_mult = 1.0 + amount * 0.5 * (rng() - 0.5);
        for (let k = 0; k < 4; k++) {
            rules[d + k] = rules[s + k] * freq_mult;
            rules[d + 4 + k] = rules[s + 4 + k] + amount * (rng() * 2.0 - 1.0);
        }
    }
}

let seed_value = 0;

function randomizeBase() {
    randomRule(BASE_SLOT, makeRng(seed_value, STREAM_BASE));
}

function deriveTargets() {
    for (let c = 0; c < MAX_COHORTS; c++) {
        mutatedRule(cohortSlot(c), BASE_SLOT, 1.0, makeRng(seed_value, streamCohort(c)));
    }
}

//------------------------------------ OUTPUT ----------------------------------

// Same layout as `rules`: 4 planes x 20 columns x 65 rows = 5200 floats
const MATRIX_COLUMNS = 2 * N_CENTERS;  // 20
const matrix = new JitterMatrix(FLOATS_PER_VEC4, "float32", MATRIX_COLUMNS, N_SLOTS);

// Copy the rules into the matrix and send it to jit.gl.texture @name cohort_rules.
// Called by every op that changes the rules, and only by those.
function output() {
    matrix.copyarraytomatrix(rules);
    outlet(0, "jit_matrix", matrix.name);
}

// Every op that changes the rules ends here: send them and refresh the view.
function rulesChanged() {
    output();
    mgraphics.redraw();
}

//------------------------------------ VIEW ------------------------------------

const VIEW = {
    pad: 6,
    label_w: 20,        // row labels (B, 0, 8, 16, ...)
    header_h: 14,       // center indices above the heatmap
    base_gap: 3,        // gap between the base row and the cohort rows
    detail_w: 300,      // numeric detail panel on the right
    line_h: 15,
    font: "Arial",
    font_size: 10,
};

// Color scale: values are mapped to [-1, 1] by type, then to a diverging map
const FREQ_RANGE = 3.0;   // base freqs in [-3, 3]
const AMP_RANGE = 1.5;    // base amps in [-1, 1], targets up to +-2

const COLOR_BG = [0.11, 0.11, 0.12, 1.0];
const COLOR_ZERO = [0.16, 0.16, 0.17];
const COLOR_NEG = [0.25, 0.55, 1.0];
const COLOR_POS = [1.0, 0.55, 0.15];
const COLOR_TEXT = [0.85, 0.85, 0.85, 1.0];
const COLOR_DIM = [0.5, 0.5, 0.5, 1.0];
const COLOR_INACTIVE = [0.11, 0.11, 0.12, 0.7];   // overlay on rows >= n_cohorts
const COLOR_SELECT = [1.0, 1.0, 1.0, 1.0];

let selected_slot = BASE_SLOT;

function valueColor(v, range) {
    let t = Math.max(-1, Math.min(1, v / range));
    const c = t < 0 ? COLOR_NEG : COLOR_POS;
    t = Math.pow(Math.abs(t), 0.7);
    return [
        COLOR_ZERO[0] + (c[0] - COLOR_ZERO[0]) * t,
        COLOR_ZERO[1] + (c[1] - COLOR_ZERO[1]) * t,
        COLOR_ZERO[2] + (c[2] - COLOR_ZERO[2]) * t,
    ];
}

// Geometry derived from the box size
function heatmapRect() {
    const size = mgraphics.size;
    const x = VIEW.pad + VIEW.label_w;
    const y = VIEW.pad + VIEW.header_h;
    const w = Math.max(FLOATS_PER_RULE, size[0] - x - VIEW.detail_w - 2 * VIEW.pad);
    const h = Math.max(N_SLOTS, size[1] - y - VIEW.pad);
    const row_h = (h - VIEW.base_gap) / N_SLOTS;
    return { x: x, y: y, w: w, h: h, col_w: w / FLOATS_PER_RULE, row_h: row_h };
}

function rowY(hm, slot) {
    return hm.y + slot * hm.row_h + (slot > BASE_SLOT ? VIEW.base_gap : 0);
}

function rowAtY(hm, y) {
    if (y < hm.y + hm.row_h + VIEW.base_gap * 0.5) return BASE_SLOT;
    return clampInt((y - hm.y - VIEW.base_gap) / hm.row_h, 1, N_SLOTS - 1);
}

function slotLabel(slot) {
    return slot === BASE_SLOT ? "base" : "cohort " + (slot - 1);
}

function isInactive(slot) {
    return slot > BASE_SLOT && slot - 1 >= n_cohorts_value;
}

function setColor(c) {
    mgraphics.set_source_rgba(c[0], c[1], c[2], c.length > 3 ? c[3] : 1.0);
}

function textRight(str, x_right, y) {
    const w = mgraphics.text_measure(str)[0];
    mgraphics.move_to(x_right - w, y);
    mgraphics.show_text(str);
}

function textLeft(str, x, y) {
    mgraphics.move_to(x, y);
    mgraphics.show_text(str);
}

function paintHeatmap(hm) {
    // cells
    for (let slot = 0; slot < N_SLOTS; slot++) {
        const y = rowY(hm, slot);
        const o = ruleOffset(slot);
        for (let col = 0; col < FLOATS_PER_RULE; col++) {
            const is_amp = (col % FLOATS_PER_CENTER) >= FLOATS_PER_VEC4;
            setColor(valueColor(rules[o + col], is_amp ? AMP_RANGE : FREQ_RANGE));
            mgraphics.rectangle(hm.x + col * hm.col_w, y, hm.col_w + 0.5, hm.row_h + 0.5);
            mgraphics.fill();
        }
    }

    // separators between centers
    setColor(COLOR_BG);
    for (let i = 1; i < N_CENTERS; i++) {
        mgraphics.rectangle(hm.x + i * FLOATS_PER_CENTER * hm.col_w - 0.5, hm.y, 1, hm.h);
        mgraphics.fill();
    }

    // inactive cohorts (>= n_cohorts) dimmed
    if (n_cohorts_value < MAX_COHORTS) {
        const y0 = rowY(hm, cohortSlot(n_cohorts_value));
        setColor(COLOR_INACTIVE);
        mgraphics.rectangle(hm.x, y0, hm.w, hm.y + hm.h - y0);
        mgraphics.fill();
    }

    // selection outline
    setColor(COLOR_SELECT);
    mgraphics.set_line_width(1);
    mgraphics.rectangle(hm.x - 0.5, rowY(hm, selected_slot) - 0.5, hm.w + 1, hm.row_h + 1);
    mgraphics.stroke();

    // center indices
    setColor(COLOR_DIM);
    for (let i = 0; i < N_CENTERS; i++) {
        textLeft(String(i), hm.x + i * FLOATS_PER_CENTER * hm.col_w + 2, hm.y - 4);
    }

    // row labels: base, every 8th cohort, and the selected row
    for (let slot = 0; slot < N_SLOTS; slot++) {
        const c = slot - 1;
        const show = slot === BASE_SLOT || c % 8 === 0 || slot === selected_slot;
        if (!show) continue;
        setColor(slot === selected_slot ? COLOR_TEXT : COLOR_DIM);
        const label = slot === BASE_SLOT ? "B" : String(c);
        textRight(label, hm.x - 3, rowY(hm, slot) + hm.row_h * 0.5 + 3.5);
    }
}

function paintDetail(hm) {
    const x0 = hm.x + hm.w + VIEW.pad + 4;
    const num_w = 33;                     // one number column
    const idx_w = 16;
    const gap = 12;                       // between freq and amp blocks
    const freq_x = x0 + idx_w;
    const amp_x = freq_x + 4 * num_w + gap;
    let y = VIEW.pad + 10;

    // title
    setColor(COLOR_TEXT);
    let title = slotLabel(selected_slot);
    if (isInactive(selected_slot)) title += "  (inactive)";
    textLeft(title, x0, y);
    setColor(COLOR_DIM);
    textRight("seed " + seed_value + "   n_cohorts " + n_cohorts_value, amp_x + 4 * num_w, y);
    y += VIEW.line_h + 2;

    // column headers
    const comps = ["x", "y", "z", "w"];
    textLeft("freq", freq_x, y);
    textLeft("amp", amp_x, y);
    y += VIEW.line_h - 3;
    for (let k = 0; k < 4; k++) {
        textRight(comps[k], freq_x + (k + 1) * num_w, y);
        textRight(comps[k], amp_x + (k + 1) * num_w, y);
    }
    y += VIEW.line_h;

    // one line per center, each number on a swatch of its heatmap color
    for (let i = 0; i < N_CENTERS; i++) {
        const o = centerOffset(selected_slot, i);
        setColor(COLOR_DIM);
        textLeft(String(i), x0, y);
        for (let k = 0; k < 4; k++) {
            const blocks = [[freq_x, rules[o + k], FREQ_RANGE], [amp_x, rules[o + 4 + k], AMP_RANGE]];
            for (const [bx, v, range] of blocks) {
                const cx = bx + k * num_w;
                setColor(valueColor(v, range));
                mgraphics.rectangle(cx + 2, y - VIEW.line_h + 4, num_w - 2, VIEW.line_h - 1);
                mgraphics.fill();
                setColor(COLOR_TEXT);
                textRight(v.toFixed(2), cx + num_w - 2, y);
            }
        }
        y += VIEW.line_h;
    }
}

function paint() {
    const size = mgraphics.size;
    setColor(COLOR_BG);
    mgraphics.rectangle(0, 0, size[0], size[1]);
    mgraphics.fill();

    mgraphics.select_font_face(VIEW.font);
    mgraphics.set_font_size(VIEW.font_size);

    const hm = heatmapRect();
    paintHeatmap(hm);
    paintDetail(hm);
}

// Click (or drag) in the heatmap or its labels selects a row
function selectAt(x, y) {
    const hm = heatmapRect();
    if (x > hm.x + hm.w) return;
    const slot = rowAtY(hm, y);
    if (slot !== selected_slot) {
        selected_slot = slot;
        mgraphics.redraw();
    }
}

function onclick(x, y) {
    selectAt(x, y);
}

function ondrag(x, y) {
    selectAt(x, y);
}

function onresize() {
    mgraphics.redraw();
}

//------------------------------------ MESSAGES --------------------------------

// seed <int>: seed used by randomize and derive when no seed is given
function seed(v) {
    seed_value = Math.floor(v);
    mgraphics.redraw();
}

// randomize [seed]: new random base rule. The optional seed also becomes the current seed.
function randomize(s) {
    if (s !== undefined) seed_value = Math.floor(s);
    randomizeBase();
    rulesChanged();
}

// derive [seed]: every cohort target (all 64 slots) = base mutated at full scale,
// each cohort with its own random stream. The live amount is the shader's `mutation`.
function derive(s) {
    if (s !== undefined) seed_value = Math.floor(s);
    deriveTargets();
    rulesChanged();
}

// bang: resend the current rules (e.g. after rebuilding the texture)
function bang() {
    output();
}

// n_cohorts <int>: number of active cohorts (1..64). Does not touch the rules:
// all 64 target slots are always kept, unused ones are ignored by the shaders.
function n_cohorts(v) {
    n_cohorts_value = clampInt(v, 1, MAX_COHORTS);
    mgraphics.redraw();
}

// dump [cohort]: post the base rule, or a cohort's target rule, to the console
function dump(cohort) {
    let slot = BASE_SLOT;
    let label = "base";
    if (cohort !== undefined) {
        const c = clampInt(cohort, 0, MAX_COHORTS - 1);
        slot = cohortSlot(c);
        label = "cohort " + c;
    }
    post("rules.js — " + label + " (n_cohorts " + n_cohorts_value + ", seed " + seed_value + ")\n");
    for (let i = 0; i < N_CENTERS; i++) {
        const o = centerOffset(slot, i);
        const f = Array.from(rules.subarray(o, o + 4), (x) => x.toFixed(3));
        const a = Array.from(rules.subarray(o + 4, o + 8), (x) => x.toFixed(3));
        post("  " + i + "  freq " + f.join(" ") + "  amp " + a.join(" ") + "\n");
    }
}

//------------------------------------ INIT ------------------------------------

// Never start with all-zero rules (zero force in the shader).
// Filled while the script loads; sent on loadbang, once the outlet is connected.
randomizeBase();
deriveTargets();

function loadbang() {
    output();
}
