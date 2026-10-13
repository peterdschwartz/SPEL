module elm_initializeMod

   use nc_io, only : io_constants, io_inputs
   !#USE_START

   implicit None

   public :: elm_init

contains

   subroutine elm_init(nsets, dt, yr, bounds)
      ! Read the input sites and, for nsets > 1, duplicate them nsets times.
      ! There is always a single clump spanning the whole process, so the
      ! filters (and the parallelism inside each routine) cover every set.
      use duplicateMod, only: duplicate_sets, scale_bounds
      use ReadWriteMod, only: read_elmtypes
      use FUTConstantsMod, only: read_constants
      use nc_io, only: spel_nsets
      use elm_varctl
      use filterMod
      use decompMod, only: get_proc_bounds, get_clump_bounds, procinfo, clumps
      use decompMod, only: bounds_type
      use timeInfoMod
      use pftvarcon
      use decompInitMod
      use domainMod

      implicit none

      integer, intent(in)  :: nsets
      real*8, intent(in) :: dt
      integer, intent(in) :: yr
      type(bounds_type), intent(out) :: bounds
      integer :: begp, endp, begc, endc, begg, endg, begl, endl, begt, endt
      real*8, allocatable  :: icemask_grc(:)
      type(bounds_type) :: bounds_clump

      if (nsets < 1) stop "(elm_init) nsets must be >= 1"
      ! nc_io allocates subgrid arrays nsets times larger and replicates
      ! the file data into every set when reading
      spel_nsets = nsets

      call read_constants(io_constants)
      !#SPEL_IO

      ! bounds now hold one set (the input file); offset the subgrid indices
      ! of the copies and extend the bounds over all sets
      if (nsets > 1) then
         print *, "Duplicating input sites", nsets, "times"
         call duplicate_sets(nsets, bounds)
         call scale_bounds(nsets, bounds)
      end if

      begp = bounds%begp; endp = bounds%endp
      begc = bounds%begc; endc = bounds%endc
      begg = bounds%begg; endg = bounds%endg
      begl = bounds%begl; endl = bounds%endl
      begt = bounds%begt; endt = bounds%endt
      print *, "begp, endp", begp, endp
      print *, "begc, endc", begc, endc
      print *, "begg, endg", begg, endg
      print *, "begl, endl", begl, endl
      print *, "begt, endt", begt, endt

      allocate (icemask_grc(begg:endg)); icemask_grc(:) = 0.d0
      call init_clumps(bounds)
      call get_clump_bounds(1, bounds_clump)
      call allocFilters()
#ifndef _OPENACC
      ! OpenACC builds set the filters in main, once the data is on the device
      call setFilters(bounds_clump, icemask_grc)
#endif

      !#LAKECON

      !#VAR_INIT_START

   end subroutine elm_init

   subroutine init_clumps(bounds_proc)
      ! A single clump spanning the process bounds
      use decompMod, only: bounds_type, clumps, procinfo, get_clump_bounds
      use decompMod, only: clump_pproc
      use filterMod, only: filter, filter_inactive_and_active
      type(bounds_type), intent(in) :: bounds_proc
      !! Locals
      type(bounds_type) :: bounds_clump

      integer :: nc, delta, nclumps
      integer :: num_g, num_l, num_t, num_c, num_p

      num_g = bounds_proc%endg - bounds_proc%begg + 1
      num_l = bounds_proc%endl - bounds_proc%begl + 1
      num_t = bounds_proc%endt - bounds_proc%begt + 1
      num_c = bounds_proc%endc - bounds_proc%begc + 1
      num_p = bounds_proc%endp - bounds_proc%begp + 1
      nclumps = 1
      clump_pproc = nclumps

      allocate (clumps(nclumps), procinfo%cid(clump_pproc))
      allocate (filter(nclumps), filter_inactive_and_active(nclumps))
      procinfo%nclumps = clump_pproc
      do nc = 1, nclumps
         procinfo%cid(nc) = nc
      end do
      procinfo%ncells = num_g
      procinfo%ntunits = num_t
      procinfo%nlunits = num_l
      procinfo%ncols = num_c
      procinfo%npfts = num_p
      procinfo%nCohorts = 0
      procinfo%begg = bounds_proc%begg
      procinfo%begt = bounds_proc%begt
      procinfo%begl = bounds_proc%begl
      procinfo%begc = bounds_proc%begc
      procinfo%begp = bounds_proc%begp
      procinfo%begCohort = 1
      procinfo%endg = bounds_proc%endg
      procinfo%endt = bounds_proc%endt
      procinfo%endl = bounds_proc%endl
      procinfo%endc = bounds_proc%endc
      procinfo%endp = bounds_proc%endp
      procinfo%endCohort = 0

      do nc = 1, nclumps
         clumps(nc)%begp = bounds_proc%begp
         clumps(nc)%endp = bounds_proc%endp
         clumps(nc)%begc = bounds_proc%begc
         clumps(nc)%endc = bounds_proc%endc
         clumps(nc)%begl = bounds_proc%begl
         clumps(nc)%endl = bounds_proc%endl
         clumps(nc)%begt = bounds_proc%begt
         clumps(nc)%endt = bounds_proc%endt
         clumps(nc)%begg = bounds_proc%begg
         clumps(nc)%endg = bounds_proc%endg
         clumps(nc)%begCohort = bounds_proc%begCohort
         clumps(nc)%endCohort = bounds_proc%endCohort
      end do

      call get_clump_bounds(1, bounds_clump)
      print *, "(init clumps) bounds_clump"
      print *, "g: ", bounds_clump%begg, bounds_clump%endg
      print *, "l: ", bounds_clump%begl, bounds_clump%endl
      print *, "t: ", bounds_clump%begt, bounds_clump%endt
      print *, "c: ", bounds_clump%begc, bounds_clump%endc
      print *, "p: ", bounds_clump%begp, bounds_clump%endp

   end subroutine init_clumps

end module elm_initializeMod
