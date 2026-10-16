module filterMod

#include "shr_assert.h"

  !-----------------------------------------------------------------------
  ! !DESCRIPTION:
  ! Module of filters used for processing columns and pfts of particular
  ! types, including lake, non-lake, urban, soil, snow, non-snow, and
  ! naturally-vegetated patches.
  !
  ! !USES:
  !
  ! !PUBLIC TYPES:
  implicit none
  save
  private
  !
  type clumpfilter
     integer, pointer :: testc(:)      ! nat-vegetated (present) filter (pfts)
     integer :: num_testc              ! number of pfts in nat-vegetated filter
  end type clumpfilter
  public clumpfilter

  ! This is the standard set of filters, which should be used in most places in the code.
  ! These filters only include 'active' points.
  type(clumpfilter), allocatable, public :: filter(:)
  !
  public allocFilters            ! allocate memory for filters
  public setFilters              ! set filters

  private allocFiltersOneGroup  ! allocate memory for one group of filters
  private setFiltersOneGroup    ! set one group of filters
  !
  !-----------------------------------------------------------------------

contains

  !------------------------------------------------------------------------
  subroutine allocFilters()
    !
    ! !DESCRIPTION:
    ! Allocate CLM filters.
    !
    ! !REVISION HISTORY:
    ! Created by Bill Sacks
    !------------------------------------------------------------------------

    call allocFiltersOneGroup(filter)
    call allocFiltersOneGroup(filter_inactive_and_active)

  end subroutine allocFilters

  !------------------------------------------------------------------------
  subroutine allocFiltersOneGroup(this_filter)
    !
    ! !DESCRIPTION:
    ! Allocate CLM filters, for one group of filters.
    !
    ! !USES:
    !
    ! !ARGUMENTS:
    type(clumpfilter), intent(inout), allocatable :: this_filter(:)  ! the filter to allocate
    !
    ! LOCAL VARAIBLES:
    integer :: nc          ! clump index
    integer :: nclumps     ! total number of clumps on this processor
    integer :: ier         ! error status
    !------------------------------------------------------------------------

    ! Determine clump variables for this processor


    ier = 0
    if( .not. allocated(this_filter)) then
       allocate(this_filter(nclumps), stat=ier)
    end if
    if (ier /= 0) then
       write(iulog,*) 'allocFiltersOneGroup(): allocation error for clumpsfilters'
       call endrun(msg=errMsg(__FILE__, __LINE__))
    end if

    ! Loop over clumps on this processor

!$OMP PARALLEL DO PRIVATE (nc,bounds)
    do nc = 1, nclumps
       allocate(this_filter(nc)%testc(1:nc))
    end do
!$OMP END PARALLEL DO

  end subroutine allocFiltersOneGroup

  !------------------------------------------------------------------------
  subroutine setFilters(bounds, icemask_grc)
    !
    ! !DESCRIPTION:
    ! Set CLM filters.
    !
    ! !ARGUMENTS:
    type(bounds_type) , intent(in) :: bounds  
    real(r8)          , intent(in) :: icemask_grc( bounds%begg: ) ! ice sheet grid coverage mask [gridcell]
    !------------------------------------------------------------------------

    call setFiltersOneGroup(bounds, &
         filter_inactive_and_active, include_inactive = .true., &
         icemask_grc = icemask_grc(bounds%begg:bounds%endg))
  end subroutine setFilters


  !------------------------------------------------------------------------
  subroutine setFiltersOneGroup(bounds, this_filter, include_inactive, icemask_grc )
    !
    ! !DESCRIPTION:
    ! Set CLM filters for one group of filters.
    !
    ! "Standard" filters only include active points. However, this routine can be used to set
    ! alternative filters that also apply over inactive points, by setting include_inactive =
    ! .true.
    !
    ! !USES:
    !
    ! !ARGUMENTS:
    type(bounds_type) , intent(in)    :: bounds  
    type(clumpfilter) , intent(inout) :: this_filter(:)              ! the group of filters to set
    logical           , intent(in)    :: include_inactive            ! whether inactive points should be included in the filters
    real(r8)          , intent(in)    :: icemask_grc( bounds%begg: ) ! ice sheet grid coverage mask [gridcell]
    !
    ! LOCAL VARAIBLES:
    integer :: nc          ! clump index
    integer :: c,l,p       ! column, landunit, pft indices
    integer :: fl          ! lake filter index
    integer :: fnl,fnlu    ! non-lake filter index
    integer :: fs          ! soil filter index
    integer :: fc, fpc     ! crop and perennial crop filter index
    integer :: fnc         ! non-crop filter index
    integer :: f, fn       ! general indices
    integer :: g           !gridcell index
    integer :: t           !topounit index
    !------------------------------------------------------------------------
    nc = bounds%clump_index

    ! ------------------------------------------------------------------
    ! Each filter below is built in two true-parallel OpenACC passes:
    !  (1) a "reduction" pass that counts how many points satisfy each
    !      predicate (gives num_xxx, independent of execution order), and
    !  (2) a "scatter" pass that uses an atomic-capture counter to claim a
    !      unique slot in the output filter array for each point that
    !      satisfies the predicate (the array *contents* - the set of
    !      points in the filter - do not depend on scatter order, only
    !      set membership matters for how these filters are subsequently
    !      used).
    ! ------------------------------------------------------------------

    ! Create lake and non-lake filters at column-level

    fl = 0
    fnl = 0
    !$acc parallel loop independent gang vector default(present) private(t,l) reduction(+:fl,fnl)
    do c = bounds%begc,bounds%endc
      end do
    this_filter(nc)%num_testc = fl


    ! Note: snow filters are reconstructed each time step in
    ! LakeHydrology and SnowHydrology.
    ! Note: the nolakeurban_barep/nolakeurban_vegp filters are also
    ! dynamic (they depend on the snow-dependent frac_veg_nosno flag) and
    ! are reconstructed each time step by setExposedvegpFilters, below.

  end subroutine setFiltersOneGroup


end module filterMod

