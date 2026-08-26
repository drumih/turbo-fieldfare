# Procedural Terminal 3D Engine Plan

## Goal

Build a procedural 3D endless runner rendered entirely with terminal symbols.
The engine and game are written in C17 with no runtime assets and no external
runtime dependencies. Geometry, animation, materials, physics, levels, and
rendering are all described in code.

The first proof of concept is a spinning torus. It must be rendered as a normal
indexed triangle mesh through the same generic pipeline that later renders the
runner, track, and obstacles. The project must not contain a torus-specific
rendering path.

## Product constraints

- C17 engine and game.
- Command-line executable for macOS and Linux terminals.
- POSIX platform layer using `termios`, `poll`, `ioctl`, `clock_gettime`, and
  `write`.
- No ncurses, Notcurses, SDL, OpenGL, Metal, Lua, ECS framework, or asset
  loader in the initial game.
- No OBJ, glTF, textures, configuration files, or resource directory.
- All geometry and animation are procedural.
- Deterministic level generation from a numeric seed.
- No heap allocation during the steady-state frame loop.
- One buffered terminal write per full frame where practical.
- ASCII fallback when the terminal or font cannot display the selected Unicode
  symbols safely.

The intended technical description is:

> A procedural 3D endless runner rendered entirely with terminal symbols. It is
> written in pure C with no dependencies and no assets; everything from
> geometry and animation to physics and levels is generated in code.

## Development sources and single-file distribution

Development must use normal, modular source files. The one-file version is a
generated release artifact, not the canonical source and not a file edited by
hand. This is an amalgamation build, similar to single-file distributions used
by established C projects.

Canonical development layout:

```text
src/
  main.c
  math.c
  math.h
  memory.c
  memory.h
  mesh.c
  mesh.h
  render.c
  render.h
  glyph.c
  glyph.h
  terminal.c
  terminal.h
  world.c
  world.h
  game.c
  game.h

tests/
  test_math.c
  test_mesh.c
  test_render.c
  test_glyph.c
  test_world.c

tools/
  amalgamate.py

dist/
  term3d.c       # generated; never edited manually
```

The initial torus milestone should use fewer modules if some of these files do
not yet have real content. Empty `world`, `game`, physics, or animation modules
must not be created in advance merely to reserve the names.

Development build:

```bash
cc -std=c17 -O3 -c src/math.c
cc -std=c17 -O3 -c src/mesh.c
cc -std=c17 -O3 -c src/render.c
cc -std=c17 -O3 -c src/glyph.c
cc -std=c17 -O3 -c src/terminal.c
cc -std=c17 -O3 -c src/main.c
cc math.o mesh.o render.o glyph.o terminal.o main.o -lm -o term3d
```

Release generation and build:

```bash
python3 tools/amalgamate.py src/main.c --output dist/term3d.c
cc -std=c17 -O3 dist/term3d.c -lm -o term3d
```

The amalgamator recursively expands project-local quoted includes, includes
each project header or source once, preserves system includes, and emits `#line`
directives so diagnostics still name the canonical source file:

```c
#line 1 "src/render.c"
```

Amalgamation-specific rules:

- Prefix externally visible and internal file-scope names with `t3d_` and a
  module name where useful.
- Do not define identically named `static` functions in separate `.c` files;
  they collide when combined into one translation unit.
- Prefix project macros and `#undef` temporary implementation macros.
- Do not rely on include order or on macros leaking between modules.
- Never edit `dist/term3d.c` directly.
- Build and test both modular and amalgamated forms in CI.
- Compare a deterministic headless frame checksum from both builds.

Suggested targets:

```text
make              modular development build
make test         modular tests
make amalgamate   regenerate dist/term3d.c
make dist-test    compile and test the generated source
make release      produce dist/term3d.c and the release executable
```

## Architecture

```text
fixed-step game update
    |-- input actions
    |-- procedural animation
    |-- simple physics and collision
    `-- deterministic track generation
                 |
                 v
          transforms and meshes
                 |
                 v
       generic indexed-mesh renderer
       transform -> clip -> project
       -> cull -> triangle rasterize
                 |
                 v
       terminal subcell sample target
       inverse depth + shade + colour
                 |
                 v
             glyph resolver
       ASCII / shade / half / quadrant / Braille
                 |
                 v
          packed terminal cells
                 |
                 v
       ANSI presenter -> buffered write()
```

The renderer consumes immutable meshes, transforms, cameras, and materials. It
must not know whether a transform came from animation, physics, procedural
generation, or input. The terminal presenter consumes resolved cells and must
not know anything about triangles or 3D math.

## Core data

```c
typedef struct { float x, y; } T3D_Vec2;
typedef struct { float x, y, z; } T3D_Vec3;
typedef struct { float x, y, z, w; } T3D_Vec4;
typedef struct { float m[16]; } T3D_Mat4;

typedef struct {
    T3D_Vec3 position;
    T3D_Vec3 normal;
} T3D_Vertex;

typedef struct {
    T3D_Vertex *vertices;
    uint32_t *indices;
    uint32_t vertex_count;
    uint32_t index_count;
} T3D_Mesh;

typedef struct {
    T3D_Vec3 position;
    T3D_Vec3 rotation;
    T3D_Vec3 scale;
} T3D_Transform;

typedef struct {
    const T3D_Mesh *mesh;
    T3D_Mat4 model;
    uint32_t material_id;
} T3D_RenderInstance;
```

Euler rotation is sufficient for the first spinning torus and procedural
runner limbs. Add quaternions only when composed rotations or interpolation
make them necessary.

Transformed vertices preserve the fields needed for clipping and later
perspective-correct interpolation:

```c
typedef struct {
    T3D_Vec4 clip_position;
    float shade;
} T3D_ClipVertex;

typedef struct {
    float x;
    float y;
    float inv_w;
    float shade_over_w;
} T3D_ScreenVertex;
```

## Procedural torus

Generate the indexed torus once during initialization. Trigonometry must not be
performed once per torus vertex per frame.

```c
static T3D_Vertex t3d_make_torus_vertex(
    float major_radius,
    float minor_radius,
    float u,
    float v)
{
    const float cu = cosf(u);
    const float su = sinf(u);
    const float cv = cosf(v);
    const float sv = sinf(v);
    const float ring = major_radius + minor_radius * cv;

    return (T3D_Vertex) {
        .position = {
            ring * cu,
            minor_radius * sv,
            ring * su
        },
        .normal = {
            cv * cu,
            sv,
            cv * su
        }
    };
}
```

Connect adjacent rings with wrapped indexed triangles:

```c
for (uint32_t i = 0; i < major_segments; ++i) {
    for (uint32_t j = 0; j < minor_segments; ++j) {
        const uint32_t i1 = (i + 1) % major_segments;
        const uint32_t j1 = (j + 1) % minor_segments;

        const uint32_t a = i  * minor_segments + j;
        const uint32_t b = i1 * minor_segments + j;
        const uint32_t c = i1 * minor_segments + j1;
        const uint32_t d = i  * minor_segments + j1;

        *index++ = a; *index++ = b; *index++ = c;
        *index++ = a; *index++ = c; *index++ = d;
    }
}
```

Start with `64 x 24` segments: 1,536 vertices and 3,072 triangles. The same
renderer must also draw a procedurally generated cube before the torus
milestone is accepted; this proves that the implementation is a generic
renderer rather than a disguised donut algorithm.

## Symbol-native sample target

Terminal glyphs such as quadrants and Braille encode multiple spatial samples
inside one cell. The renderer therefore writes depth, coverage, shade, and
colour into a small subcell target. These samples exist only to resolve one
terminal glyph; they are not a separate image or texture pipeline.

```c
typedef struct {
    uint16_t cell_cols;
    uint16_t cell_rows;
    uint8_t samples_x;
    uint8_t samples_y;
    uint16_t width;
    uint16_t height;

    float *inv_depth;
    uint8_t *shade;
    uint32_t *colour;
} T3D_SampleTarget;
```

Allocate capacity for the largest built-in mode, Braille at `2 x 4`, once.
Less detailed modes use smaller active dimensions without reallocating.

At `120 x 40` terminal cells, Braille mode contains only 38,400 samples, so a
single-threaded CPU rasterizer is sufficient until profiling proves otherwise.

## Rendering pipeline

For every frame:

1. Clear the active inverse-depth and shading arrays.
2. Build model, view, and projection matrices.
3. Transform each unique mesh vertex once.
4. Transform its normal and calculate directional lighting.
5. Assemble indexed triangles.
6. Clip triangles against the near plane.
7. Perform perspective division.
8. Correct projection for terminal-cell aspect ratio.
9. Cull back-facing or degenerate triangles.
10. Rasterize with incremental edge functions.
11. Interpolate inverse depth and shade.
12. Depth-test every active subcell sample.
13. Resolve samples into packed terminal cells.
14. Encode either a full frame or changed runs into the output arena.
15. Present the completed byte stream with a partial-write-safe loop.

Initial lighting:

```c
float diffuse = fmaxf(0.0f, t3d_dot3(normal, light_direction));
float shade = 0.15f + 0.85f * diffuse;
```

Use inverse depth so zero represents an empty sample and larger values are
closer:

```c
if (inv_depth > target->inv_depth[index]) {
    target->inv_depth[index] = inv_depth;
    target->shade[index] = t3d_quantize_shade(shade);
    target->colour[index] = colour;
}
```

Rasterize with edge functions and a consistent top-left fill rule:

```c
static inline int64_t t3d_edge(
    int32_t ax, int32_t ay,
    int32_t bx, int32_t by,
    int32_t px, int32_t py)
{
    return (int64_t)(px - ax) * (by - ay)
         - (int64_t)(py - ay) * (bx - ax);
}
```

Transform and clip in floating point, then use a fixed-point screen coordinate
representation for the inner raster loop. Once the initial edge values are
known, advance them across rows with additions rather than recalculating the
full expression for every sample.

The first torus may be placed wholly in front of the near plane, but near-plane
clipping is required before movable cameras or arbitrary procedural scenes.
Do not clamp vertices to the near plane. Full six-plane homogeneous clipping
can follow when off-screen meshes make it necessary.

## Glyph modes

Built-in modes:

| Mode | Samples per cell | Symbols |
|---|---:|---|
| ASCII | `1 x 1` | ` .:-=+*#%@` |
| Dense ASCII | `1 x 1` | `.,-~:;=!*#$@` |
| Shade | `1 x 1` | ` `, `░`, `▒`, `▓`, `█` |
| Half block | `1 x 2` | ` `, `▀`, `▄`, `█` |
| Quadrant | `2 x 2` | sixteen block masks |
| Braille | `2 x 4` | U+2800 through U+28FF |

Braille bit order is not row-major:

```c
static const uint8_t t3d_braille_bit[4][2] = {
    { 1u << 0, 1u << 3 },
    { 1u << 1, 1u << 4 },
    { 1u << 2, 1u << 5 },
    { 1u << 6, 1u << 7 }
};
```

Precompute all built-in glyphs as UTF-8 during initialization. Braille becomes
a direct table lookup:

```c
glyph = braille_utf8[coverage_mask];
```

Call `setlocale(LC_CTYPE, "")`, validate custom ramp code points with
`wcwidth() == 1`, and fall back to ASCII for unsupported locales or glyphs.
Reject combining marks, variation selectors, zero-width joiners, and emoji in
custom ramps.

## Terminal cells and presentation

Resolve the sample target into packed logical cells before producing terminal
bytes. A 64-bit cell should contain a glyph-table index, foreground colour,
background colour, and flags. Equality then requires one integer comparison.

Maintain `next` and `shown` cell grids and support:

```text
--present full
--present diff
--present auto
```

- `full` homes the cursor and encodes the complete frame.
- `diff` groups adjacent changed cells into horizontal runs and emits one cursor
  movement per run.
- `auto` estimates both byte costs and emits the smaller representation.

Do not assume differential output is faster. A rotating object or moving camera
may change enough cells that a complete frame is smaller than many cursor
commands.

Never use `printf`, `putchar`, `fflush`, `snprintf`, or dynamic string growth in
the cell loop. Build a contiguous byte stream and handle partial writes:

```c
static bool t3d_write_all(int fd, const void *data, size_t size)
{
    const uint8_t *p = data;

    while (size != 0) {
        const ssize_t n = write(fd, p, size);

        if (n > 0) {
            p += (size_t)n;
            size -= (size_t)n;
        } else if (n < 0 && errno == EINTR) {
            continue;
        } else {
            return false;
        }
    }

    return true;
}
```

The terminal backend must:

- Verify that input and output are TTYs for interactive mode.
- Save and restore the exact original terminal attributes.
- Use the alternate screen and hide the cursor.
- Read dimensions with `ioctl(TIOCGWINSZ)`.
- Use `poll` for input and frame waiting.
- Let signal handlers set only `volatile sig_atomic_t` flags.
- Handle `SIGWINCH` in the main loop.
- Restore terminal state after normal exit, Ctrl-C, termination, or errors.
- Avoid writing a newline after the final row.
- Avoid accidental scrolling from the bottom-right cell.

## Memory policy

Use one live backing allocation where practical. Divide it into aligned regions
for long-lived and frame buffers:

```c
size_t bytes =
    vertex_bytes +
    index_bytes +
    transformed_bytes +
    depth_bytes +
    shade_bytes +
    colour_bytes +
    cell_bytes * 2 +
    output_bytes;

uint8_t *memory = malloc(bytes);
uint8_t *cursor = memory;

vertices = t3d_take(&cursor, vertex_bytes, _Alignof(T3D_Vertex));
indices  = t3d_take(&cursor, index_bytes, _Alignof(uint32_t));
depth    = t3d_take(&cursor, depth_bytes, _Alignof(float));
cells_a  = t3d_take(&cursor, cell_bytes, _Alignof(uint64_t));
cells_b  = t3d_take(&cursor, cell_bytes, _Alignof(uint64_t));
output   = t3d_take(&cursor, output_bytes, 1);
```

Allowed allocations:

- Initial backing block.
- A complete replacement block after terminal resize.

Forbidden during steady-state frames:

- `malloc`
- `calloc`
- `realloc`
- `free`
- growing containers or strings

On resize, allocate the entire replacement first, swap only after success, and
then free the old block. Keep the old renderer alive if replacement allocation
fails. A debug allocation counter must verify zero frame-loop allocations.

## Fixed update and procedural animation

Use a fixed simulation timestep independent of terminal presentation:

```c
const double fixed_dt = 1.0 / 60.0;
double accumulator = 0.0;
double previous = t3d_monotonic_seconds();

while (!quit_requested) {
    const double now = t3d_monotonic_seconds();
    double elapsed = now - previous;
    previous = now;

    if (elapsed > 0.25)
        elapsed = 0.25;

    accumulator += elapsed;
    t3d_poll_input(&input);

    while (accumulator >= fixed_dt) {
        t3d_game_update(&game, (float)fixed_dt);
        accumulator -= fixed_dt;
    }

    t3d_render_scene(&renderer, &game.scene);
    t3d_resolve_glyphs(&renderer, glyph_mode);
    t3d_present(&terminal, renderer.cells);
    t3d_wait_until_next_frame();
}
```

Advance an absolute monotonic deadline rather than sleeping for a complete frame
duration after rendering.

The torus animation is initially just code:

```c
demo.rotation.x += 0.7f * dt;
demo.rotation.y += 1.1f * dt;
```

## Procedural geometry and character

Required code-generated primitives:

```c
T3D_Mesh t3d_make_cube(...);
T3D_Mesh t3d_make_box(...);
T3D_Mesh t3d_make_torus(...);
T3D_Mesh t3d_make_cylinder(...);
T3D_Mesh t3d_make_sphere(...);
T3D_Mesh t3d_make_capsule(...);
T3D_Mesh t3d_make_ramp(...);
T3D_Mesh t3d_make_arch(...);
T3D_Mesh t3d_make_track(...);
```

Assemble the runner from primitive instances rather than creating a single
special mesh:

```c
typedef struct {
    T3D_Transform body;
    T3D_Transform head;
    T3D_Transform arm_l;
    T3D_Transform arm_r;
    T3D_Transform leg_l;
    T3D_Transform leg_r;
} T3D_RunnerPose;
```

Initial running animation:

```c
const float phase = run_time * run_speed;

pose.arm_l.rotation.x =  sinf(phase) * 0.8f;
pose.arm_r.rotation.x = -sinf(phase) * 0.8f;
pose.leg_l.rotation.x = -sinf(phase) * 0.9f;
pose.leg_r.rotation.x =  sinf(phase) * 0.9f;
pose.body.position.y = fabsf(sinf(phase * 2.0f)) * 0.04f;
```

Game animation states:

```text
idle
running
jumping
falling
sliding
crashed
```

Each state is a small C function that calculates a pose. Do not add a skeletal
animation framework unless procedural poses become insufficient.

## Deterministic endless track

Use a small deterministic PRNG:

```c
typedef struct {
    uint32_t state;
} T3D_Rng;

static uint32_t t3d_rng_next(T3D_Rng *rng)
{
    uint32_t x = rng->state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return rng->state = x;
}
```

Generate track segments ahead of the camera and recycle segments that pass
behind it:

```c
typedef enum {
    T3D_SEGMENT_EMPTY,
    T3D_SEGMENT_BARRIER,
    T3D_SEGMENT_GAP,
    T3D_SEGMENT_ARCH,
    T3D_SEGMENT_COINS,
    T3D_SEGMENT_MOVING_OBSTACLE
} T3D_SegmentType;
```

The same seed and input sequence must generate the same game state, level, and
score. This enables deterministic tests, replays, and shareable challenges
without storing level files.

## Minimal physics

Implement only the behavior required by a lane-based endless runner:

```c
typedef struct {
    T3D_Vec3 position;
    T3D_Vec3 velocity;
    int lane;
    bool on_ground;
} T3D_RunnerBody;
```

Required behavior:

- gravity
- jump impulse
- smooth lane interpolation
- ground collision
- player capsule versus obstacle box collision
- trigger volumes for collectibles
- gradual forward-speed increase

For tens of active objects, a direct pair scan is acceptable. Do not implement
torque, joints, stacked rigid bodies, arbitrary convex collision, or a general
physics engine before the game demonstrates a need.

## Fast inverse square root experiment

Keep the Quake-style fast inverse square root as an optional implementation and
benchmark, not as the default contract:

```c
#ifdef T3D_QUAKE_RSQRT
#define t3d_rsqrt(x) t3d_fast_rsqrt(x)
#else
#define t3d_rsqrt(x) (1.0f / sqrtf(x))
#endif
```

Procedural primitives should generate analytic unit normals, rotation preserves
their length, and fixed directions are normalized once. The engine should avoid
normalization rather than force the bit hack into the frame loop. Enable it by
default only if a real target benchmark shows an improvement without unacceptable
error.

## CLI

```text
term3d
  --glyph ascii|dense|shade|half|quadrant|braille
  --ramp STRING
  --colour mono|16|256|truecolor
  --present full|diff|auto
  --fps N
  --size COLSxROWS
  --aspect RATIO
  --segments MAJORxMINOR
  --seed N
  --frames N
  --headless
  --benchmark
  --stats
  --dump-frame PATH
  --glyph-test
```

Initial controls:

```text
q or Esc    quit
1 through 5 select glyph mode
c           cycle colour mode
Space       pause
Arrow keys  rotate while the torus demo is paused
```

Later game controls should use discrete left, right, jump, and slide actions.
Traditional terminal input does not reliably report key releases, so game
movement must not require precise press/release state.

## Milestones

### 0. Terminal and glyph probe

Deliver terminal entry/restoration, resize, raw input, UTF-8 tables, a glyph
test grid, and one full-frame buffered write.

Gate:

- Normal exit and Ctrl-C restore the terminal.
- Every selected glyph occupies one column.
- ASCII fallback works.
- Sanitizers find no error across small and large terminal sizes.

### 1. Generic torus renderer

Deliver math, generic indexed meshes, procedural torus and cube, camera,
backface culling, fixed-point edge rasterization, inverse depth, Lambert
lighting, ASCII/shade modes, and fixed-step rotation.

Gate:

- Correct occlusion and no cracks between adjacent triangles.
- No torus-specific branch in the renderer.
- The same path draws a cube.
- Zero steady-state allocations.
- A fixed headless frame produces a stable logical-cell checksum.

### 2. Symbol-native rendering

Deliver half-block, quadrant, and Braille resolution, ordered dithering,
runtime mode switching, Unicode fallback, and aspect correction.

Gate:

- All 16 quadrant and all 256 Braille masks map correctly.
- Depth is independent for every active subcell.
- Switching modes or resizing leaves no stale samples.

### 3. Fast presenter

Deliver packed front/back cells, full and changed-run encoders, automatic byte
cost selection, colour-state caching, and presenter telemetry.

Gate:

- No per-cell stdio call.
- Normally one buffered `write` per full frame.
- Interrupted and partial writes are correct.
- Full and differential presentation are chosen from measurements.

### 4. Robust procedural scene

Deliver near-plane clipping, multiple instances, movable camera, perspective-
correct attributes, frustum rejection, and the complete primitive set.

Gate:

- Triangles crossing the camera plane do not explode.
- Opaque output does not depend on triangle submission order.
- Random off-screen triangles never write outside the buffers.

### 5. Procedural runner

Deliver the primitive-composed runner, run/jump/slide poses, lane movement,
simple collision, score, deterministic track segments, and seeded replay.

Gate:

- Recorded input reproduces identical state and score.
- Rendering contains no game or physics behavior.
- No external file is required to start a complete game.
- The torus remains as a regression/demo mode.

### 6. Amalgamated release

Deliver the generator, checked-in `dist/term3d.c`, modular and amalgamated CI
builds, and a one-command user build.

Gate:

- Modular and amalgamated builds pass the same tests.
- Both builds produce the same deterministic headless checksum.
- `dist/term3d.c` contains no unresolved project-local include.
- `cc -std=c17 -O3 dist/term3d.c -lm -o term3d` succeeds on supported systems.

## Validation and performance reporting

Measure independently:

```text
simulation time
vertex-transform time
clipping time
triangle-raster time
glyph-resolution time
ANSI-encoding time
terminal-write time
bytes per frame
changed cells and runs
missed frame deadlines
allocations per frame
```

Required benchmark modes:

```bash
term3d --headless --benchmark --frames 10000
term3d --present full --frames 1000 --stats
term3d --present diff --frames 1000 --stats
term3d --present auto --frames 1000 --stats
```

Suggested builds:

```make
CFLAGS_DEBUG = -std=c17 -O0 -g3 -Wall -Wextra -Wshadow -Wconversion \
               -fsanitize=address,undefined
CFLAGS_RELEASE = -std=c17 -O3 -DNDEBUG -flto -Wall -Wextra
LDLIBS = -lm
```

Use `-march=native` only for local benchmark builds. Do not enable
`-ffast-math`, add SIMD intrinsics, or add raster threads until profiling shows
that scalar rendering rather than terminal output is the bottleneck.

## Final acceptance criteria

```text
canonical development sources: modular C files
release source:                one generated C file
external runtime dependencies: zero
runtime assets:                zero
procedural geometry:           all
procedural animation:          all
procedural levels:             all
steady-state allocations:      zero
normal full-frame writes:      one
deterministic seed and replay:  supported
headless benchmark:             supported
```

## Explicit non-goals until proven necessary

- Runtime asset loading.
- Textures or conventional pixel output.
- General ECS framework.
- Scene graph.
- Lua or another scripting VM.
- Skeletal animation framework.
- General rigid-body physics.
- Multithreaded or SIMD rasterization.
- GPU renderer.
- Windows terminal backend.
- Custom build system.

The plan intentionally keeps reusable boundaries around rendering, glyph
resolution, terminal presentation, and fixed-step game updates while rejecting
speculative subsystems. Future work should add complexity only when a measured
game requirement crosses the current design's stated ceiling.
