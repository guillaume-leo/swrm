# SWRM — Simulation Evolution Plan

## Context

SWRM is a real-time particle simulation built in **Max/MSP (Jitter)**. The particles live on the GPU:

- `code/tf.simulation.jxs`: shader loaded by `jit.gl.tf` (transform feedback). Each vertex is a particle. Its state is a `vec4` (`xy` = position, `zw` = velocity), updated every frame.
- `code/mesh.shader.jxs`: particle render shader (`jit.gl.mesh` drawing points). It writes color to output 0 and velocity to output 1 of the `trail` node.
- `patchers/main.maxpat`: the patch. It runs the trail loop (additive deposit, persistence, diffusion) that produces the `trail_tex` texture read back by the simulation. **This loop already exists and stays in the patch. This plan does not touch it.**

The goal is to reproduce the behavior of **Fluoddity** (check fluoddity_shader.glsl provided in this repo). Particles deposit their velocity into a field (the trails), sample it with two sensors (left/right), and react according to a **rule**. A rule is a 4D → 4D function built from 10 Fourier centers (a `vec4` frequency plus a `vec4` amplitude). It outputs a force, which changes velocity, and a strafe, which moves the particle directly. Each evaluation runs twice, once in normal and once in mirrored coordinates, to cancel any clockwise/counterclockwise bias. Particles are split into **cohorts** by index, and each cohort has its own rule.

Unlike Fluoddity, rules are **not generated in the simulation shader**. They are generated and edited **on the CPU side, in a `v8ui` script** (`code/rules.js`). The script holds a **base rule** plus one **target rule** per cohort, writes them **directly into a `jit.matrix`**, which the patch turns into the `cohort_rules` texture (`v8ui → jit.gl.texture`), and **displays them** in the patch (a heatmap of every rule plus the detail of the selected one), with the edit actions available by clicking. No Max `array` objects are involved. This makes rules editable per cohort (randomize, copy, mutate, load a custom rule) and easy to store in presets and interpolate between presets.

**Mutation amount is applied in the shader**, as a live uniform `mutation`. Each particle uses `mix(base, target[cohort], mutation)`: at `mutation = 0` every cohort follows the base rule, at `mutation = 1` every cohort follows its own target. This reproduces Fluoddity's `mutation_scale`: if a target is the base mutated at full scale (additive noise on amplitudes, multiplicative noise on frequencies), the mix at `m` is exactly the base mutated at scale `m`. Values above 1 extrapolate past the targets, which is allowed.

Two use cases are targeted:

- **Exploration**: edit cohort rules, highlight a cohort to observe it, save what's interesting as presets.
- **Live**: rules are recalled from presets, with interpolated transitions between them.

The GPU side is identical in both modes: the texture is re-uploaded only when the rules change. The difference lives in the patch's interface.

## Target Architecture

```
  edit ops (randomize, derive,          presets
  copy, mutate, load)                   (store / recall / interpolate)
          │                                   │
          ▼                                   ▼
   ┌─────────────────────────────────────────────────┐
   │ v8ui script code/rules.js                       │
   │ base rule + one target rule per cohort          │
   │ heatmap + selected rule detail, click actions   │
   │ writes them into a JitterMatrix 4 float32 20 65 │
   └────────────────────────┬────────────────────────┘
                            ▼  (on change: jit_matrix <name>)
                 jit.gl.texture @name cohort_rules
                            │
          ┌─────────────────┴─────────────────┐
          ▼                                   ▼
 tf.simulation.jxs (jit.gl.tf)        mesh.shader.jxs (render)
  reads trail_tex + cohort_rules,      cohort, highlight
 mix(base, target, mutation)
          │
          ▼
  particle state ──▶ render ──▶ trail_tex (patch loop)
```

### Rule Data Layout

A rule is a flat list of **80 floats**, in the same order as the texture:

```
[ freq0.xyzw, amp0.xyzw, freq1.xyzw, amp1.xyzw, ..., freq9.xyzw, amp9.xyzw ]
```

- One base rule, plus one target rule per cohort (up to 64 cohorts). In the script, all rules live in a single `Float32Array(65 * 80)`: slot 0 is the base, cohort `c` is slot `c + 1` (offset `(c + 1) * 80`).
- The base has its own slot so that every cohort, including cohort 0, can diverge from it, and editing one cohort never moves the others.
- Because a rule is a plain list of numbers, a preset is just the base plus the set of target rule lists (and the `mutation` value). Interpolating between two presets is an element-wise linear interpolation. Fourier rules vary continuously with their coefficients, so interpolated rules should produce continuous behavior changes.

### `cohort_rules` Texture Format

- The script owns a `JitterMatrix(4, "float32", 20, 65)`, filled in one call with `copyarraytomatrix(rules)` (the `Float32Array` has exactly the matrix layout), and sends `jit_matrix <name>` out of its outlet. The patch feeds it to `jit.gl.texture @name cohort_rules`: `float32`, no filtering, `@rectangle 0`.
- Row 0 = base rule. Row `c + 1` = target rule of cohort `c`.
- The matrix always has **65 rows**, whatever `n_cohorts` is: unused rows are ignored by the shaders. This avoids resizing the matrix and texture when the cohort count changes.
- In a row, cell `2i` holds the `frequency` of center `i`, cell `2i+1` its `amplitude` (4 planes = `xyzw`).
- Shader read, once per particle: `mix(texelFetch(cohort_rules, ivec2(2*i + k, 0), 0), texelFetch(cohort_rules, ivec2(2*i + k, cohort + 1), 0), mutation)`.
- Particle cohort: `floor(n_cohorts * gl_VertexID / particle_count)`, using the same formula in the simulation and render shaders.
- Size: 20 × 65 cells (5200 floats). Rewriting and re-uploading it every frame during an interpolation is cheap. Changing `mutation` needs no upload at all.

### Edit Operations (CPU side, in the script)

Each operation is a message to the `v8ui` object (or a click in its interface) and applies to the base, to one cohort's target, or to all cohort targets.

| op | Effect |
|---|---|
| `randomize` | New random rule from a seed (Fluoddity's `generate_random_centers`: frequencies biased toward low values in ~[-3, 3], amplitudes in [-1, 1]). On the base, or on cohort target(s) |
| `derive` | Target(s) = base mutated at **full scale** (Fluoddity's `mutate_rule` with scale 1: additive noise on amplitudes, multiplicative noise on frequencies). This reproduces Fluoddity, where all cohorts are variants of a single rule; the actual amount is then set live by the `mutation` uniform |
| `copy` | Exact copy of `source` (base or a cohort target) |
| `mutate` | In-place nudge of target(s), scaled by an amount given in the message |
| `load` | Set the base or a target from a custom rule (e.g. imported from Fluoddity's JSON config) |

The random generation does not need to be bit-exact with Fluoddity's GPU hash, but it should use a seeded PRNG so a seed always gives the same rule.

## Steps

Each step has a checkbox. It is checked when the step is committed, with the commit hash appended: `- [x] **S1. …** (`abc1234`)`.

The work is split into three tracks:

- **Shader track** (`S*`): shader files in `code/` (`*.jxs`). Done by the AI.
- **Script track** (`J*`): the `v8ui` script `code/rules.js`. Done by the AI.
- **Patch track** (`P*`): `patchers/main.maxpat`. Done **manually** by the user. The AI never edits the patch; it may only describe what to add.

The tracks meet through the interfaces defined below. Shader and script steps can be written before the patch exists, but they can only be tested in Max once the matching patch steps are done.

### Shader ↔ Patch Interface

Everything the patch must provide to the shaders:

| Name | Type | Used by | Provided by patch as |
|---|---|---|---|
| `trail_tex` | sampler2D | simulation | existing trail loop (unchanged) |
| `cohort_rules` | sampler2D, 20 × 65, float32, no filtering | simulation | `jit.gl.texture @name cohort_rules`, fed by the `v8ui` script (see texture format above) |
| `mutation` | float (`0` = all cohorts follow the base, `1` = each cohort follows its target) | simulation | `param mutation …` |
| `n_cohorts` | float | simulation, render | `param n_cohorts …` |
| `particle_count` | float | simulation, render | `param particle_count …` |
| `frame` | float | simulation | frame counter, `0` = reset |
| `sensor_gain`, `sensor_angle`, `sensor_distance` | float | simulation | `param …` |
| `global_force_mult`, `axial_force`, `lateral_force`, `strafe_power` | float | simulation | `param …` |
| `hazard_rate` | float | simulation | `param …` |
| `drag`, `aspect`, `boundary_mode` | float | simulation | existing `param …` messages |
| `highlight_cohort` | float (`-1` = none) | render | `param highlight_cohort …` |

### Script ↔ Patch Interface

The `v8ui @filename rules.js` object:

- **Inlet**: messages for the edit ops (`randomize`, `derive`, `copy`, `mutate`, `load`) and settings (`n_cohorts`, `seed`). The mutation amount is not a script setting: it is the shader's `mutation` uniform. Exact message signatures are defined in the script step that adds them.
- **Outlet 0**: `jit_matrix <name>`, sent whenever the rules change, to connect to `jit.gl.texture @name cohort_rules`.

### V1 — Simulation driven by the rule texture

**Shader track**

- [x] **S1. Rewrite `code/tf.simulation.jxs`**: (`3e037f1`)
  - declare the uniforms listed in the interface (simulation column);
  - two sensors reading `trail_tex.xy` (deposited velocities), keeping the current coordinate convention (aspect, `uv.y` flip);
  - the cohort's rule is read from `cohort_rules` once per particle into local arrays (`texelFetch(cohort_rules, ivec2(2*i + k, cohort), 0)`), then passed to `fourier_noise` for both the base and mirror terms (`calculate_entity_behavior` with mirror symmetry);
  - integration `vel = vel*drag + force`, then `pos += vel + strafe*strafe_power`, then the existing `handleBounds`;
  - reset when `frame == 0` or according to `hazard_rate`.
- [x] **S2. Remove Fluoddity's world-scale constants** (`SQRT_WORLD_SIZE`, `38.855`): they are plain fixed multipliers, folded into the defaults of `sensor_gain`, `sensor_distance` and `global_force_mult` (calibrated by eye once the patch is wired). `/400` and `/20` (with the `SQRT_WORLD_SIZE` factor: `/200` and `/10`) stay as internal normalizations. (`3e037f1`)
- [x] **S3. Mutation in the shader**: add the `mutation` uniform to `code/tf.simulation.jxs`; `load_rule` reads the base (row 0) and the cohort's target (row `cohort + 1`) and mixes them with `mutation`. (`d48fc8f`)

**Script track**  
  
Resources:  
https://docs.cycling74.com/apiref/js/jittermatrix/  
https://docs.cycling74.com/apiref/js/jitterobject/

- [x] **J1. Rule storage** in `code/rules.js`: base + targets in a `Float32Array(65 * 80)`, plus `n_cohorts`. (`9e599ba`)
- [x] **J2. `randomize` and `derive` ops** with a seeded PRNG (see Edit Operations): `randomize` on the base, `derive` on all cohort targets. (`ea64fb9`)
- [ ] **J3. Matrix output**: write the rules into the `JitterMatrix 4 float32 20 65` (`copyarraytomatrix`) and send `jit_matrix <name>` out of outlet 0, only when the rules change.
- [ ] **J4. Rule view** (`mgraphics`): heatmap of the 65 rules × 80 floats (rows beyond `n_cohorts` greyed out), click to select a row, numeric detail of the selected rule (10 centers, `freq.xyzw` / `amp.xyzw`). Redraw only when the rules or the selection change.

**Patch track (manual)**

- [ ] **P1. Rule texture**: add `v8ui @filename rules.js` (sized to show the view) and connect its outlet to `jit.gl.texture @name cohort_rules @type float32 @rectangle 0` with filtering off.
- [ ] **P2. Simulation wiring**: pass both textures to `jit.gl.tf` (check that `@texture trail_tex cohort_rules` works), add the `param …` messages for the new uniforms (including `mutation`), add a frame counter with reset.

**V1 test**: requires S1–S3, J1–J4, P1 and P2.

### V2 — Per-cohort editing and highlight

**Shader track**

- [ ] **S4. `code/mesh.shader.jxs`**:
  - same cohort formula as the simulation, with `n_cohorts`, `particle_count`, `highlight_cohort` uniforms;
  - a new `layout(location = 2)` output for the highlight;
  - one color per cohort instead of the `id < 500000` test.

**Script track**

- [ ] **J5. `copy` / `paste` and `mutate` ops**, and every op can target the base, a specific cohort, or all cohorts. Available both as messages and as click actions in the view (buttons drawn in the `v8ui`; keyboard copy/paste if `v8ui` receives key events).

**Patch track (manual)**

- [ ] **P3. Render wiring**: set the `trail` node to `@capture 3`, route the highlight output, add the `n_cohorts`, `particle_count`, `highlight_cohort` messages to the mesh shader. The cohort selector and edit-op controls live in the `v8ui` view.

**V2 test**: requires S4, J5 and P3.

### V3 — Presets, custom rules, live

**Script track**

- [ ] **J6. Preset support**: expose the full set of rules to Max's preset system (e.g. `getvalueof` / `setvalueof` for `pattr`), and interpolate between two stored rule sets, rewriting the matrix on each interpolation step.
- [ ] **J7. `load` op** for custom rules (e.g. importing Fluoddity's JSON config format).

**Patch track (manual)**

- [ ] **P4. Presets**: store and recall the rules (through the script) plus the physics parameters and `mutation` in Max's preset system.
- [ ] **P5. Live mode**: interface to trigger interpolated transitions between presets.

## Extra informations

- **output resolution**: 1920x1080 
- **Maximum number of cohorts**: 64
- **Target particle count**: 1 to 4 millions

## AI behavior:
- NEVER do multiple task in a row, always ask for validation before going to the next point
- NEVER edit `patchers/main.maxpat`: patch steps are done manually by the user
- When a step is committed, check its box in this file and append the short commit hash
