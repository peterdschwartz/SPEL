module elm_initializeMod

   use nc_io, only : io_constants, io_inputs
   !#USE_START

   implicit None

   public :: elm_init
   public :: correct_physical_properties

contains

   subroutine elm_init(nsets, pproc_input, dt, yr, bounds)

      !use elm_varsur, only: wt_lunit, urban_valid, wt_glc_mec
      use duplicateMod, only: duplicate_clumps, duplicate_weights
      ! use ReadConstantsMod, only: readConstants
      ! use InitalizeParametersMod
      use ReadWriteMod, only: read_elmtypes
      use FUTConstantsMod, only: read_constants
      use elm_varctl
      use filterMod
      use decompMod, only: get_proc_bounds, get_clump_bounds, procinfo, clumps
      use decompMod, only: bounds_type, clump_pproc, nclumps
      use timeInfoMod
      use pftvarcon
      use decompInitMod
      use domainMod
      ! use UnitTestAllocatorMod

      implicit none

      integer, intent(in)  :: nsets, pproc_input
      real*8, intent(in) :: dt
      integer, intent(in) :: yr
      type(bounds_type), intent(out) :: bounds
      integer :: errc
      integer :: nc
      integer :: begp, endp, begc, endc, begg, endg, begl, endl, begt, endt
      integer :: i, j
      integer, parameter :: number_of_sites = 42
      integer :: ni, nj = 1
      integer :: c, l, g, t
      integer, allocatable :: amask(:)
      real*8, allocatable  :: icemask_grc(:)
      real(r8), allocatable :: h2osno_col(:)
      real(r8), allocatable :: snow_depth_col(:)
      real(r8), allocatable :: dummy_2d_arr(:, :)
      type(bounds_type) :: bounds_clump

      ! convert number use input of sets of sites to total number of sites
      ni = nsets*number_of_sites
      clump_pproc = pproc_input

      call read_constants(io_constants)
      call read_elmtypes(io_inputs, bounds)
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
      call init_clumps(clump_pproc, bounds)
      call get_clump_bounds(1, bounds_clump)
      call allocFilters()
      call setFilters(bounds_clump, icemask_grc)

      !#LAKECON

      !allocate (amask(ni*nj)); amask(:) = 1
      !call decompInit_lnd_simple(ni, nj, amask)

      ! Allocate surface grid dynamic memory (just gridcell bounds dependent)
      !NOTE: wt_lunit, urban_valid need to be updated to support
      !  multiple topounits if desired
      call get_proc_bounds(begg=begg, endg=endg)

      if (nsets > 1) then
         print *, nsets, "sets requested, making a total # of gridcells of ", ni
         call duplicate_weights(number_of_sites, ni)
      end if

      !print *, "calling decompinit_clumps"
      !call decompInit_clumps()

      !#VAR_INIT_START

      if (nsets > 1) then
         print *, "Duplicating physical properties"
         call duplicate_clumps(mode=0, unique_sites=number_of_sites, num_sites=ni)
         call correct_physical_properties(number_of_sites, ni)
      end if
    !!read for first clumps only

      call get_clump_bounds(1, bounds_clump)

      ! Only duplicate if desired simulation is greater than
      ! number of gridcells in input data file
      if (nsets > 1) then
         print *, "Duplicating the remaining variables"
         call duplicate_clumps(mode=1, unique_sites=number_of_sites, num_sites=ni)
      end if


   end subroutine elm_init

   subroutine correct_physical_properties(unique_sites, num_sites)
      use landunit_varcon, only: max_lunit
      use GridcellType, only: grc_pp
      use VegetationType, only: veg_pp
      use ColumnType, only: col_pp
      use LandunitType, only: lun_pp
      use TopounitType, only: top_pp
      use landunit_varcon, only: max_lunit
      use elm_varcon, only: ispval
      use decompMod, only: bounds_type, procinfo, nclumps, get_clump_bounds
      implicit none
      integer, intent(in) :: num_sites, unique_sites
      integer :: nc, nc_copy, i, stride, j
      type(bounds_type)  :: bounds
      integer :: begt, endt, begl, endl, begc, endc, begp, endp
      integer :: total_landunit_indices, total_columns
      integer :: total_lu, total_pft
      integer :: l, c, p, l_copy, c_copy, p_copy

      total_landunit_indices = 0
      !get total_landunit_indices for the set of unique sites
      nc = unique_sites
      do j = 1, max_lunit
         i = top_pp%landunit_indices(j, nc)
         if (i /= ispval) then
            total_landunit_indices = max(total_landunit_indices, i)
         end if
      end do

      do nc = unique_sites + 1, num_sites
         nc_copy = mod(nc - 1, unique_sites) + 1
         call get_clump_bounds(nc, bounds)
         begt = bounds%begt; endt = bounds%endt
         begl = bounds%begl; endl = bounds%endl

         grc_pp%gindex(nc) = nc
         grc_pp%topi(nc) = begt
         grc_pp%topf(nc) = endt
         grc_pp%ntopounits(nc) = grc_pp%ntopounits(nc_copy) !endt-begt+1

         lun_pp%gridcell(begl:endl) = nc
         !NOTE: this won't work for multiple Topounits
         lun_pp%topounit(begl:endl) = nc

         do j = 1, max_lunit
            if (top_pp%landunit_indices(j, 1) == ispval) cycle
            top_pp%landunit_indices(j, nc) = top_pp%landunit_indices(j, nc_copy) + total_landunit_indices
         end do
      end do

      ! get total number of cols and landunits for the set
      call get_clump_bounds(unique_sites, bounds)
      total_columns = bounds%endc
      total_lu = bounds%endl
      total_pft = bounds%endp

      do nc = unique_sites + 1, num_sites
         nc_copy = mod(nc - 1, unique_sites) + 1
         call get_clump_bounds(nc, bounds)
         begl = bounds%begl; endl = bounds%endl
         begc = bounds%begc; endc = bounds%endc
         begp = bounds%begp; endp = bounds%endp

         do l = begl, endl
            l_copy = l - total_lu
            lun_pp%coli(l) = lun_pp%coli(l_copy) + total_columns
            lun_pp%colf(l) = lun_pp%colf(l_copy) + total_columns
            lun_pp%pfti(l) = lun_pp%pfti(l_copy) + total_pft
            lun_pp%pftf(l) = lun_pp%pftf(l_copy) + total_pft
         end do
         col_pp%gridcell(begc:endc) = nc
         col_pp%topounit(begc:endc) = nc !Need To change to allow multi topo

         do c = begc, endc
            c_copy = c - total_columns
            col_pp%landunit(c) = col_pp%landunit(c_copy) + total_lu
            col_pp%pfti(c) = col_pp%pfti(c_copy) + total_pft
            col_pp%pftf(c) = col_pp%pftf(c_copy) + total_pft
         end do
         veg_pp%gridcell(begp:endp) = nc
         veg_pp%topounit(begp:endp) = nc !Need to change
         do p = begp, endp
            p_copy = p - total_pft
            veg_pp%landunit(p) = veg_pp%landunit(p_copy) + total_lu
            veg_pp%column(p) = veg_pp%column(p_copy) + total_columns
         end do
      end do

   end subroutine

   subroutine init_clumps(clump_pproc, bounds_proc)
      use decompMod, only: bounds_type, clumps, procinfo, get_clump_bounds
      use filterMod, only: filter, filter_inactive_and_active
      integer, intent(in) :: clump_pproc
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
      nclumps = clump_pproc
      print *, "(init_clumps) Using nclumps ", nclumps
      if (nclumps > 1) then
         stop "Need to verify clumps > 1"
      end if

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
