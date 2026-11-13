# SWRM

A real-time GPU particle swarm built in **Max/MSP (Jitter)**. Millions of particles move across a shared field: each one leaves a trail of its velocity, senses the trails around it, and reacts according to a **rule** owned by its cohort. Rules are edited, stored and blended on the CPU side, so the swarm can be explored by hand or played live through presets.

*Inspired by [Fluoddity](#credits).*

## How it works

Each frame, the simulation runs in a loop:

1. **Simulate**: `jit.gl.tf` (transform feedback) updates every particle on the GPU. A particle's state is a `vec4`: `xy` = position, `zw` = velocity.
2. **Sense**: each particle samples the trail texture with two sensors (left / right) placed ahead of it along its direction of travel.
3. **React**: its cohort's rule turns these readings into a **force** (which changes velocity) and a **strafe** (which moves the particle directly). The rule is evaluated twice, in normal and mirrored coordinates, to cancel any left/right bias.
4. **Render and deposit**: particles are drawn as points into the `trail` node, which writes their velocity into a texture. The patch's trail loop (additive deposit, persistence, diffusion) produces `trail_tex`, which the next frame reads.

```
  rule arrays (CPU) ──▶ cohort_rules texture
                              │
  trail_tex ──▶ tf.simulation.jxs ──▶ particle state ──▶ mesh.shader.jxs ──▶ trail loop ──┐
      ▲                                                                                    │
      └────────────────────────────────────────────────────────────────────────────────────┘
```

### Rules and cohorts

- Particles are split into **cohorts** by index: `cohort = floor(n_cohorts * id / particle_count)`, up to 64 cohorts.
- A **rule** is a 4D → 4D Fourier function made of 10 centers, each with a `vec4` frequency and a `vec4` amplitude: **80 floats** per rule.
- Rules live in Max `array` objects and are uploaded as the `cohort_rules` texture (`jit.matrix 4 float32 20 N`, one row per cohort, cell `2i` = frequency `i`, cell `2i+1` = amplitude `i`).
- Since a rule is a plain list of numbers, a preset is just a set of lists, and moving between two presets is an element-wise interpolation.

## Repository layout

| Path | Contents |
|---|---|
| [swrm.maxproj](swrm.maxproj) | Max project file. Open this. |
| [patchers/main.maxpat](patchers/main.maxpat) | Main patch: `jit.world`, trail loop, simulation and render wiring. |
| [code/tf.simulation.jxs](code/tf.simulation.jxs) | Simulation shader (loaded by `jit.gl.tf`): sensors, rule evaluation, integration, bounds, resets. |
| [code/mesh.shader.jxs](code/mesh.shader.jxs) | Render shader (loaded by `jit.gl.mesh`): draws particles, outputs color and velocity. |
| [media/mire.jpg](media/mire.jpg) | Test image. |
| [fluoddity_shader.glsl](fluoddity_shader.glsl) | Reference compute shader, kept for comparison (see [Credits](#credits)). |
| [PLAN.md](PLAN.md) | Development plan and step tracking. |

## Getting started

1. Use a recent version of **Max** with the GL3 Jitter engine (the patch relies on `jit.gl.tf` and `v8`).
2. Open `swrm.maxproj`. The project adds `code/`, `patchers/` and `media/` to the search path.
3. The `main.maxpat` patcher opens and renders into a `jit.world` window (target output: 1920 × 1080).

## Simulation parameters

Set on the simulation shader with `param <name> <value>`.

| Parameter | Default | Role |
|---|---|---|
| `n_cohorts` | 1 | Number of cohorts (max 64) |
| `particle_count` | 1000000 | Total particles (target 1 to 4 million) |
| `frame` | 0 | Frame counter, `0` resets the simulation |
| `sensor_gain` | 19.43 | Multiplier on sensed trail values |
| `sensor_angle` | 0.25 | Sensor angle, in units of π |
| `sensor_distance` | 1.0 | Sensor distance ahead of the particle |
| `global_force_mult` | 1.0 | Global multiplier on force and strafe |
| `axial_force` | 1.0 | Weight of the forward component |
| `lateral_force` | 1.0 | Weight of the sideways component |
| `strafe_power` | 1.0 | Weight of strafe in the position update |
| `drag` | 0.95 | Velocity damping per frame |
| `hazard_rate` | 0.0 | Per-frame probability that a particle respawns |
| `aspect` | 1.0 | World aspect ratio |
| `boundary_mode` | 0 | `0` = wrap, `1` = bounce |

Textures are bound in this order: `@texture trail_tex cohort_rules`.

## Status

Work in progress. Progress is tracked in [PLAN.md](PLAN.md):

- **V1, rule-driven simulation**: shader done; patch side (rule arrays, randomize / derive, rule texture, wiring) to do.
- **V2, per-cohort editing and highlight**: to do.
- **V3, presets, custom rules, live interpolation**: to do.

## Credits

The particle model (velocity trails, two-sensor sensing, mirrored Fourier rules, cohorts) is inspired by **Fluoddity**. `fluoddity_shader.glsl` is kept in this repository as a reference.
