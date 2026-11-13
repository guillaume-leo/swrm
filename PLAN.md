# SWRM — Simulation Evolution Plan

## Context

SWRM is a real-time particle simulation built in **Max/MSP (Jitter)**. The particles live on the GPU:

- `code/tf.simulation.jxs`: shader loaded by `jit.gl.tf` (transform feedback). Each vertex is a particle. Its state is a `vec4` (`xy` = position, `zw` = velocity), updated every frame.
- `code/mesh.shader.jxs`: particle render shader (`jit.gl.mesh` drawing points). It writes color to output 0 and velocity to output 1 of the `trail` node.
- `patchers/main.maxpat`: the patch. It runs the trail loop (additive deposit, persistence, diffusion) that produces the `trail_tex` texture read back by the simulation. **This loop already exists and stays in the patch. This plan does not touch it.**

The goal is to reproduce the behavior of **Fluoddity** (check fluoddity_shader.glsl provided in this repo). Particles deposit their velocity into a field (the trails), sample it with two sensors (left/right), and react according to a **rule**. A rule is a 4D → 4D function built from 10 Fourier centers (a `vec4` frequency plus a `vec4` amplitude). It outputs a force, which changes velocity, and a strafe, which moves the particle directly. Each evaluation runs twice, once in normal and once in mirrored coordinates, to cancel any clockwise/counterclockwise bias. Particles are split into **cohorts** by index, and each cohort has its own rule.

Unlike Fluoddity, rules are **not computed in the simulation shader**. They are generated and edited **on the CPU side, in Max `array` objects**, then converted to a `cohort_rules` texture (`array → jit.matrix → jit.gl.texture`). This makes them editable per cohort (randomize, copy, mutate, load a custom rule) and easy to store in presets and interpolate between presets. The simulation only reads the texture.

Two use cases are targeted:

- **Exploration**: edit cohort rules, highlight a cohort to observe it, save what's interesting as presets.
- **Live**: rules are recalled from presets, with interpolated transitions between them.

The GPU side is identical in both modes: the texture is re-uploaded only when the rule arrays change. The difference lives in the patch's interface.

## Target Architecture

```
  edit ops (randomize, derive,          presets
  copy, mutate, load)                   (store / recall / interpolate)
          │                                   │
          ▼                                   ▼
   ┌─────────────────────────────────────────────────┐
   │ rule arrays (Max `array`)                       │
   │ one rule per cohort = 80 floats                 │
   └────────────────────────┬────────────────────────┘
                            ▼  (on change)
                 jit.matrix 4 float32 20 N
                            ▼
                 jit.gl.texture @name cohort_rules
                            │
          ┌─────────────────┴─────────────────┐
          ▼                                   ▼
 tf.simulation.jxs (jit.gl.tf)        mesh.shader.jxs (render)
  reads trail_tex + cohort_rules       cohort, highlight
          │
          ▼
  particle state ──▶ render ──▶ trail_tex (patch loop)
```

### Rule Data Layout

A rule is a flat list of **80 floats**, in the same order as the texture:

```
[ freq0.xyzw, amp0.xyzw, freq1.xyzw, amp1.xyzw, ..., freq9.xyzw, amp9.xyzw ]
```

- One rule per cohort, up to 64 cohorts.
- Because a rule is a plain list of numbers, a preset is just the set of cohort rule lists. Interpolating between two presets is an element-wise linear interpolation. Fourier rules vary continuously with their coefficients, so interpolated rules should produce continuous behavior changes.

### `cohort_rules` Texture Format

- `jit.matrix 4 float32 20 N` uploaded to `jit.gl.texture @name cohort_rules`: `float32`, no filtering, `@rectangle 0`.
- One row = one cohort. Cell `2i` holds the `frequency` of center `i`, cell `2i+1` its `amplitude` (4 planes = `xyzw`).
- Shader read: `texelFetch(cohort_rules, ivec2(2*i + k, cohort), 0)`.
- Particle cohort: `floor(n_cohorts * gl_VertexID / particle_count)`, using the same formula in the simulation and render shaders.
- Size: at most 20 × 64 cells (5120 floats). Re-uploading every frame during an interpolation is cheap.

### Edit Operations (CPU side)

Each operation applies to one target cohort or to all cohorts.

| op | Effect on target cohort(s) |
|---|---|
| `randomize` | New random rule from a seed (Fluoddity's `generate_random_centers`: frequencies biased toward low values in ~[-3, 3], amplitudes in [-1, 1]) |
| `derive` | Copy of `source`, then mutation. This reproduces Fluoddity, where all cohorts are variants of a single rule |
| `copy` | Exact copy of `source` |
| `mutate` | In-place mutation scaled by `mutation_scale` (Fluoddity's `mutate_rule`: additive noise on amplitudes, multiplicative noise on frequencies) |
| `load` | Set from a custom rule (e.g. imported from Fluoddity's JSON config) |

The random generation does not need to be bit-exact with Fluoddity's GPU hash, but it should use a seeded PRNG so a seed always gives the same rule.

## Steps

The work is split into two tracks:

- **Shader track** (`S*`): files in `code/` (`*.jxs`). Can be done now, by the AI.
- **Patch track** (`P*`): `patchers/main.maxpat`. Done **manually** by the user, later. The AI never edits the patch; it may only describe what to add.

The two tracks meet through the interface defined below. Shader steps can be written before the patch exists, but they can only be tested in Max once the matching patch steps are done.

### Shader ↔ Patch Interface

Everything the patch must provide to the shaders:

| Name | Type | Used by | Provided by patch as |
|---|---|---|---|
| `trail_tex` | sampler2D | simulation | existing trail loop (unchanged) |
| `cohort_rules` | sampler2D, 20 × N, float32, no filtering | simulation | `jit.gl.texture @name cohort_rules` (see texture format above) |
| `n_cohorts` | float | simulation, render | `param n_cohorts …` |
| `particle_count` | float | simulation, render | `param particle_count …` |
| `frame` | float | simulation | frame counter, `0` = reset |
| `sensor_gain`, `sensor_angle`, `sensor_distance` | float | simulation | `param …` |
| `global_force_mult`, `axial_force`, `lateral_force`, `strafe_power` | float | simulation | `param …` |
| `hazard_rate` | float | simulation | `param …` |
| `drag`, `aspect`, `boundary_mode` | float | simulation | existing `param …` messages |
| force / strafe / sensor gains (step S2) | float | simulation | `param …` |
| `highlight_cohort` | float (`-1` = none) | render | `param highlight_cohort …` |

### V1 — Simulation driven by the rule texture

**Shader track**

- **S1. Rewrite `code/tf.simulation.jxs`**:
  - declare the uniforms listed in the interface (simulation column);
  - two sensors reading `trail_tex.xy` (deposited velocities), keeping the current coordinate convention (aspect, `uv.y` flip);
  - the cohort's rule is read from `cohort_rules` once per particle into local arrays (`texelFetch(cohort_rules, ivec2(2*i + k, cohort), 0)`), then passed to `fourier_noise` for both the base and mirror terms (`calculate_entity_behavior` with mirror symmetry);
  - integration `vel = vel*drag + force`, then `pos += vel + strafe*strafe_power`, then the existing `handleBounds`;
  - reset when `frame == 0` or according to `hazard_rate`.
- **S2. Replace Fluoddity's constants with tunable gains** (`SQRT_WORLD_SIZE`, `38.855`, `/400`, `/20`) exposed as uniforms. They depend on Fluoddity's resolution and density, so they are calibrated by eye once the patch is wired.

**Patch track (manual)**

- **P1. Rule storage**: rule arrays in Max `array` objects (80 floats per cohort, up to 64 cohorts).
- **P2. `randomize` and `derive` ops** with a seeded PRNG (see Edit Operations). The logic may live in a `v8` script in `code/`; the AI can write that script on request, the user integrates it in the patch.
- **P3. Rule texture**: `array → jit.matrix 4 float32 20 N → jit.gl.texture @name cohort_rules`, re-uploaded only when the arrays change (to be validated: filling the 4-plane matrix through `v8` or `jit.fill`).
- **P4. Simulation wiring**: pass both textures to `jit.gl.tf` (check that `@texture trail_tex cohort_rules` works), add the `param …` messages for the new uniforms, add a frame counter with reset.

**V1 test**: requires S1, S2 and P1–P4.

### V2 — Per-cohort editing and highlight

**Shader track**

- **S3. `code/mesh.shader.jxs`**:
  - same cohort formula as the simulation, with `n_cohorts`, `particle_count`, `highlight_cohort` uniforms;
  - a new `layout(location = 2)` output for the highlight;
  - one color per cohort instead of the `id < 500000` test.

**Patch track (manual)**

- **P5. `copy` and `mutate` ops**, and every op can target a specific cohort or all cohorts.
- **P6. Render wiring**: set the `trail` node to `@capture 3`, route the highlight output, add the `n_cohorts`, `particle_count`, `highlight_cohort` messages to the mesh shader, and a cohort selector in the interface.

**V2 test**: requires S3, P5 and P6.

### V3 — Presets, custom rules, live (patch only)

**Patch track (manual)**

- **P7. Presets**: store and recall the full set of cohort rules (plus the physics parameters) in Max's preset system.
- **P8. `load` op** for custom rules (e.g. importing Fluoddity's JSON config format).
- **P9. Live mode**: interpolate between presets (element-wise on rule lists and parameters), re-uploading the texture on each interpolation step.

## Extra informations

- **output resolution**: 1920x1080 
- **Maximum number of cohorts**: 64
- **Target particle count**: 1 to 4 millions

## AI behavior:
- NEVER do multiple task in a row, always ask for validation before going to the next point
- NEVER edit `patchers/main.maxpat`: patch steps are done manually by the user
