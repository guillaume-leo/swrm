// rules.js — SWRM rule storage and editing (v8ui object)
//
// Holds one base rule and one target rule per cohort. A rule is 80 floats:
//   [ freq0.xyzw, amp0.xyzw, freq1.xyzw, amp1.xyzw, ..., freq9.xyzw, amp9.xyzw ]
// All rules live in a single Float32Array laid out exactly like the
// cohort_rules texture (4 planes, 20 columns, 65 rows):
//   slot 0     = base rule      (row 0)
//   slot c + 1 = target of cohort c (row c + 1)

autowatch = 1;
inlets = 1;
outlets = 1;

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

//------------------------------------ MESSAGES --------------------------------

// seed <int>: seed used by randomize and derive when no seed is given
function seed(v) {
    seed_value = Math.floor(v);
}

// randomize [seed]: new random base rule. The optional seed also becomes the current seed.
function randomize(s) {
    if (s !== undefined) seed(s);
    randomRule(BASE_SLOT, makeRng(seed_value, STREAM_BASE));
}

// derive [seed]: every cohort target (all 64 slots) = base mutated at full scale,
// each cohort with its own random stream. The live amount is the shader's `mutation`.
function derive(s) {
    if (s !== undefined) seed(s);
    for (let c = 0; c < MAX_COHORTS; c++) {
        mutatedRule(cohortSlot(c), BASE_SLOT, 1.0, makeRng(seed_value, streamCohort(c)));
    }
}

// n_cohorts <int>: number of active cohorts (1..64). Does not touch the rules:
// all 64 target slots are always kept, unused ones are ignored by the shaders.
function n_cohorts(v) {
    n_cohorts_value = clampInt(v, 1, MAX_COHORTS);
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

// Never start with all-zero rules (zero force in the shader)
randomize();
derive();
