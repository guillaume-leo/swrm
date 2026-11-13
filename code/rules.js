// rules.js — SWRM rule storage and editing (v8 object)
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

//------------------------------------ MESSAGES --------------------------------

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
    post("rules.js — " + label + " (n_cohorts " + n_cohorts_value + ")\n");
    for (let i = 0; i < N_CENTERS; i++) {
        const o = centerOffset(slot, i);
        const f = Array.from(rules.subarray(o, o + 4), (x) => x.toFixed(3));
        const a = Array.from(rules.subarray(o + 4, o + 8), (x) => x.toFixed(3));
        post("  " + i + "  freq " + f.join(" ") + "  amp " + a.join(" ") + "\n");
    }
}
