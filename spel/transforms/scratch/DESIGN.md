# DeviceStack: one arena for the routine-local arrays (GPU)

Status: design, 2026-10-25. Nothing built yet. Measure first (step 0).

## Problem

ELM's biogeophysics routines declare their work arrays as automatic locals sized by
the clump bounds or a filter count (`real(r8) :: dth(bounds%begp:bounds%endp)`,
`qin(1:num_snowc)`, `hk(1:num_hydrologyc,1:nlevgrnd)`). On the host they live on the
stack and cost nothing. In an OpenACC build every one of them that a compute region
touches needs device storage for the duration of the call: an `!$acc data create`
region (explicit or implicit) around the kernels, i.e. a device allocation and free per
array per call, plus a present-table entry per array that every kernel launch in the
routine looks up. With tens of arrays per routine, several calls per step per clump,
this is a large fixed cost that does not shrink with the problem size.

Inventory (analysis of ~/merge/E3SM HEAD, 2026-10-25, routines that contain OpenACC
directives): 246 routines, 77 of them with runtime-sized local arrays, 579 such arrays
in all (7 fixed-size). The heavy ones: `UrbanFluxes` 82, `CanopyFluxes` 60,
`LakeTemperature` 38, `SurfaceAlbedo` 34, `ch4_tran` 31, `soilwater_zengdecker2009` 27,
`BareGroundFluxes` 25, `SnowWater` 22, `SoilTemperature` 17. Shapes are of a handful of
kinds: `bounds%begc:bounds%endc` / `begp:endp` (81), `1:num_<filter>` (about 150, often
with a level dimension), and deferred-shape locals (128, `(:)`, allocated in the routine).

## Step 0: measure before building

On a pm-gpu run with `nsys profile --trace=cuda,openacc`:

- `cuda_api_sum`: counts and time of `cuMemAlloc`/`cuMemFree` (and
  `cuMemAllocHost`) per step. If they are not a visible share of the land step, this
  transform is not worth its complexity and the kernel-launch count (below) is the
  problem instead.
- `cuda_gpu_kern_sum` and the OpenACC trace: launches per step per routine; for
  `CanopyFluxes` the launches per stability iteration.
- The present-table cost shows up as host time between launches in the OpenACC
  `acc_enqueue_launch` events.

## The DeviceStack (the design; decided 2026-10-25)

One arena per run, not one per routine or module: a module `DeviceStackMod` owns a single
device-resident pool per type (`real(r8)`, `integer`, `logical`), allocated once at
initialization and large enough for every routine-local array of the run, plus a bump
pointer. Routines carve their locals out of it on entry and release them on exit, an
arena allocator with stack discipline:

```fortran
module DeviceStackMod
  real(r8), allocatable, target :: ds_r8(:)      ! the arena: host mirror + device copy
  integer , allocatable, target :: ds_i4(:)
  logical , allocatable, target :: ds_l4(:)
  integer :: ds_top_r8 = 0, ds_top_i4 = 0, ds_top_l4 = 0   ! bump pointers (words)
  !$acc declare create(ds_r8, ds_i4, ds_l4)
  !$omp threadprivate(ds_top_r8, ds_top_i4, ds_top_l4)   ! one arena region per clump thread
contains
  subroutine ds_init(n_r8, n_i4, n_l4)   ! allocate once, after the sizes are known
  integer function ds_mark_r8()          ! save the bump pointer on routine entry
  subroutine ds_release_r8(mark)         ! restore it on every exit path
  subroutine ds_alloc_r8(ptr, lb, ub)    ! ptr(lb:) => ds_r8(top+1:top+n); top = top + n; abort on overflow
  subroutine ds_alloc_r8_2d(ptr, lb1, ub1, lb2, ub2)   ! rank remap onto a contiguous slice
end module
```

In a rewritten routine every selected local `real(r8) :: dth(bounds%begp:bounds%endp)`
becomes `real(r8), pointer, contiguous :: dth(:) => null()`; the entry gets
`mark = ds_mark_r8(); call ds_alloc_r8(dth, bounds%begp, bounds%endp)`, every exit
(`return` and the final `end`) gets `call ds_release_r8(mark)`, and the body is
unchanged: the pointer carries the original bounds, so no index is rewritten. Rank-2 and
rank-3 locals use rank remapping onto a 1-D slice, legal because the slice is simply
contiguous.

Data movement: none for these arrays. The arena is present for the whole run, so the
`!$acc data create(...)`, `enter data create(...)`, `exit data delete(...)` and
`copyin`/`copyout`/`copy` clauses that name the selected locals are deleted from the
routine's directives; kernels find the slices through the arena's present-table entry.
`copy` stays only for what genuinely has to come back to the host, such as an error index
or a convergence flag. A local that host statements read or write between two kernels
(the analysis sees that: the statement lies outside every compute construct) keeps its
`update device`/`update self` directives, which work on a slice of a present array; the
host mirror is the same arena, so host code reads and writes the slice in place.

Sizing: the arena must hold the deepest live set of the run. The rewrite knows every
selected local's extent as an expression of `bounds%...` and `num_<filter>` (times level
counts), and the call tree gives the nesting of arena routines, so it emits a sizing
routine `ds_required(bounds, filter sizes) = max over root-to-leaf chains of the summed
extents`, evaluated once at initialization for the process's largest clump and the
maximum filter sizes (known after `allocFilters`/`reweight_wrapup`; filters stay within
the clump's bounds, which bound them). `ds_alloc_*` aborts on overflow with the routine
name. A first version may use the simpler upper bound "sum over all arena routines"
(about 140 MB per rank for the stride-8 case by the inventory above), then switch to the
chain maximum.

Threads: the CPU threaded build runs one clump per OpenMP thread, so the bump pointers
are `threadprivate` and each thread owns one region of the arena (`ds_init` sizes
`nthreads` regions). The GPU build runs one thread per rank. ELM has no recursion, so
the stack discipline cannot interleave.

Selection rule (per routine, per array): a non-dummy, non-`save`, non-`parameter`,
non-`target` array whose extents are not compile-time constants; not passed on whole to
an unanalyzed callee; no initializer. Fixed-size locals (7 in the inventory) stay as they
are. Deferred-shape allocatable locals that the routine allocates and frees itself (128
in the inventory) are candidates too: `allocate` becomes `ds_alloc`, `deallocate` and the
`allocated()` tests go, when the analysis shows a single allocate before the first use.

Correctness check: bit-for-bit against the untouched build on the CPU (the transform
changes storage, not arithmetic), on the crop20 and stride-8 harnesses, before any GPU
measurement; then the step-0 nsys counts before and after on the GPU.

## What SPEL already provides

- `Subroutine.record.declarations`: name, rank (`dim`), attributes, intent, the
  `VariableDecl` node with each entity's `bounds` (start/end expressions) and
  initializer; dummies by `dummy_args`.
- `record_access`: per-array read/write order (the `read_before_write` check) and whole
  array passes to callees (with the callee's analysis status).
- `spel/transforms/acc_directives.py`: the routine's `!$acc data`/`parallel` clauses and
  the arrays they name, to delete the selected arrays from `create`/`copyin` lists.
- Text edits placed by line, as the compact rewrite does (`FileEdit.before/after/subs`).

## Related: CanopyFluxes' iteration loop

The stability iteration (`do while (fnold > 0 .and. itlef < itmax)`) launches every
kernel in its body once per iteration (up to 40), over a shrinking filter. Two
independent fixes, both measurable with the step-0 profile:

1. Invert the loop: one kernel over patches with the iteration inside, each patch
   converging on its own. Everything per patch in the body is already written per
   patch; the calls over the filter inside the body (`FrictionVelocity`,
   `MoninObukIni`, `Photosynthesis`, the stomatal routines) need `!$acc routine seq`
   per-patch versions. The convergence bookkeeping (`fnold`, the new filter) disappears.
   This also removes the filter rebuild and its copies. The analysis can confirm that no
   statement in the body reads another patch's value (no reductions, no `p2c`/`c2p` in
   the loop).
2. Keep the structure, fuse the body's kernels into fewer launches (SPEL loop fusion over
   adjacent `do fp` loops with no dependence between them) and move the convergence
   reduction into the last kernel.

(1) is the real fix and is how the other land models' GPU ports handle this loop.
