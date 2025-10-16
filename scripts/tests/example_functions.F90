module test_sub_parse

  use shr_const_mod
  use constants_mod, only: nlevdecomp_full, use_fates

  use remove_mod, only : elm_fates
  real(r8), parameter :: nan = SHR_CONST_SPVAL

  integer, parameter, public :: BOUNDS_SUBGRID_GRIDCELL = 1
  integer, parameter, public :: BOUNDS_SUBGRID_TOPOUNIT = 2
  integer, parameter, public :: BOUNDS_SUBGRID_LANDUNIT = 3
  integer, parameter, public :: BOUNDS_SUBGRID_COLUMN   = 4
  integer, parameter, public :: BOUNDS_SUBGRID_PATCH    = 5
  integer, parameter, public :: BOUNDS_SUBGRID_COHORT   = 6

  !
  ! Define possible bounds levels
  integer, parameter, public :: BOUNDS_LEVEL_PROC  = 1
  integer, parameter, public :: BOUNDS_LEVEL_CLUMP = 2
  integer :: nlevdecomp
  integer,  pointer :: ft_index_bigleaf(:)                      ! array holding the pft index of each competitor
  real(r8), allocatable,target :: veg_rootc_bigleaf(:,:)        ! column-level fine-root biomas kgc/m3
  real :: array(1:10)

#ifndef DCPL
   integer :: cpp(11:11)
#else
   integer :: og(1)
   if (.true.) then 
      print*, 'what'
   end if
#endif

 type bounds_type
     ! The following variables correspond to "Local" quantities
     integer :: begg, endg                       ! beginning and ending gridcell index
     integer :: begt, endt                       ! beginning and ending topographic unit index
     integer :: begl, endl                       ! beginning and ending landunit index
     integer :: begc, endc                       ! beginning and ending column index
     integer :: begp, endp                       ! beginning and ending pft index
     integer :: begCohort, endCohort             ! beginning and ending cohort indices

  end type bounds_type
  public bounds_type

   type, public :: column_nitrogen_flux
      ! harvest fluxes
      real(r8), pointer :: hrv_deadstemn_to_prod10n(:) => null() ! dead stem N harvest mortality to 10-year product pool (gN/m2/s)
      real(r8), pointer :: hrv_deadstemn_to_prod100n(:) => null() ! dead stem N harvest mortality to 100-year product pool (gN/m2/s)
      real(r8), pointer :: m_n_to_litr_met_fire(:, :) => null() ! N from leaf, froot, xfer and storage N to litter labile N by fire (gN/m3/s)
      real(r8), pointer :: m_n_to_litr_lig_fire(:, :) => null() ! N from leaf, froot, xfer and storage N to litter labile N by fire (gN/m3/s)
      contains
         procedure, public :: Init  => col_nf_init
   end type

  public :: soil_water_retention_curve_type

  type, abstract :: soil_water_retention_curve_type
     private
   contains
     ! compute hydraulic conductivity
     procedure(soil_hk_interface), deferred :: soil_hk

     ! compute soil suction potential
     procedure(soil_suction_interface), deferred :: soil_suction

     ! compute relative saturation at which soil suction is equal to a target value
     procedure(soil_suction_inverse_interface), deferred :: soil_suction_inverse
  end type soil_water_retention_curve_type

   type(column_nitrogen_flux), target :: col_nf

  type, public :: ConcTransportType
     !! Type that points to decomposition pools for vertical transport calculations

     real(r8), pointer :: conc_ptr(:,:,:) => null()
     real(r8), pointer :: src_ptr(:,:,:)  => null()
     real(r8), pointer :: trcr_tend_ptr(:,:,:) => null()
  end type ConcTransportType
  type(ConcTransportType), public, allocatable :: transport_ptr_list(:)

   type :: prior_weights_type
     real(r8), allocatable :: pwtcol(:)
     real(r8), allocatable :: cactive(:)
   end type prior_weights_type

  type :: patch_state_updater_type

     real(r8), pointer :: pwtgcell_old(:) => null() ! old patch weights on the gridcell
     real(r8), pointer :: pwtgcell_new(:) => null()! new patch weights on the gridcell

     real(r8), pointer :: cwtgcell_old(:) => null()! old column weights on the gridcell

     ! (pwtgcell_new - pwtgcell_old) from last call to set_new_weights
     real(r8), pointer :: dwt(:) => null()

     ! (pwtgcell_old / pwtgcell_new) from last call to set_new_weights; only valid for
     ! growing patches
     real(r8), pointer :: growing_old_fraction(:) => null()

     ! (dwt / pwtgcell_new) from last call to set_new_weights; only valid for growing
     ! patches
     real(r8), pointer :: growing_new_fraction(:) => null()
   end type

   type :: clump_filter 
      integer :: num_soilc
      integer, ALLOCATABLE :: soilc(:)
   end type

   type(clump_filter), allocatable :: filter(:)

   type(test_type), public :: mytype_inst

  ! !PUBLIC MEMBER FUNCTIONS:
  public :: Tridiagonal
  interface Tridiagonal
    module procedure Tridiagonal_sr
    module procedure Tridiagonal_mr
  end interface Tridiagonal

contains
   subroutine createLitterTransportList()
      ! This subroutine creates a list that will point to the
      ! litter/som fields needed for the vertical transport
      ! calculations.

      implicit none

      integer :: ntype

      ntype = 3
      if ( use_c13 ) then
         ntype = ntype+1
      endif
      if ( use_c14 ) then
         ntype = ntype+1
      endif

      allocate(transport_ptr_list(ntype))
      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      ! C
      transport_ptr_list(1)%conc_ptr        => col_cs%decomp_cpools_vr
      transport_ptr_list(1)%src_ptr         => col_cf%decomp_cpools_sourcesink
      transport_ptr_list(1)%trcr_tend_ptr   => col_cf%decomp_cpools_transport_tendency
      ! N
      transport_ptr_list(2)%conc_ptr        => col_ns%decomp_npools_vr
      transport_ptr_list(2)%src_ptr         => col_nf%decomp_npools_sourcesink
      transport_ptr_list(2)%trcr_tend_ptr   => col_nf%decomp_npools_transport_tendency
      ! P
      transport_ptr_list(3)%conc_ptr        => col_ps%decomp_ppools_vr
      transport_ptr_list(3)%src_ptr         => col_pf%decomp_ppools_sourcesink
      transport_ptr_list(3)%trcr_tend_ptr   => col_pf%decomp_ppools_transport_tendency
      ! c13 and c14 if there
      if(use_c14 .and. use_c13) then
         !
         transport_ptr_list(4)%conc_ptr       => c13_col_cs%decomp_cpools_vr
         transport_ptr_list(4)%src_ptr        => c13_col_cf%decomp_cpools_sourcesink
         transport_ptr_list(4)%trcr_tend_ptr  => c13_col_cf%decomp_cpools_transport_tendency
         !
         transport_ptr_list(5)%conc_ptr       => c14_col_cs%decomp_cpools_vr
         transport_ptr_list(5)%src_ptr        => c14_col_cf%decomp_cpools_sourcesink
         transport_ptr_list(5)%trcr_tend_ptr  => c14_col_cf%decomp_cpools_transport_tendency
      else
         if(use_c13) then
            transport_ptr_list(4)%conc_ptr       => c13_col_cs%decomp_cpools_vr
            transport_ptr_list(4)%src_ptr        => c13_col_cf%decomp_cpools_sourcesink
            transport_ptr_list(4)%trcr_tend_ptr  => c13_col_cf%decomp_cpools_transport_tendency
         end if
         if (use_c14) then
            transport_ptr_list(4)%conc_ptr       => c14_col_cs%decomp_cpools_vr
            transport_ptr_list(4)%src_ptr        => c14_col_cf%decomp_cpools_sourcesink
            transport_ptr_list(4)%trcr_tend_ptr  => c14_col_cf%decomp_cpools_transport_tendency
         end if
      end if

   end subroutine createLitterTransportList

   subroutine allocInit(bounds)
      type(bounds_type), intent(in) :: bounds
       allocate(ft_index_bigleaf(bounds%begp:bounds%endp)); 
       ft_index_bigleaf(bounds%begp:bounds%endp) = -1
       allocate(veg_rootc_bigleaf(bounds%begp:bounds%endp,1:nlevdecomp));veg_rootc_bigleaf(bounds%begp:bounds%endp,1:nlevdecomp) = nan
   end subroutine allocInit

   subroutine allocate_filter(this, begc, endc)
      type(clump_filter) , INTENT(INOUT) :: this
      integer, intent(in) :: begc, endc
      allocate(this%soilc(1:endc-begc+1)); this%soilc(:) = -1
   end subroutine allocate_filter

function constructor(bounds) result(this)
    !
    ! !DESCRIPTION:
    ! Initialize a patch_state_updater_type object
    !
    ! !USES:
    !
    ! !ARGUMENTS:
    type(patch_state_updater_type) :: this  ! function result
    type(bounds_type), intent(in) :: bounds
    !
    ! !LOCAL VARIABLES:
    integer :: begp, endp
    integer :: begc, endc

    character(len=*), parameter :: subname = 'constructor'
    !-----------------------------------------------------------------------

    begp = bounds%begp
    endp = bounds%endp
    begc = bounds%begc
    endc = bounds%endc

    allocate(this%pwtgcell_old(begp:endp))
    allocate(this%pwtgcell_new(begp:endp))
    allocate(this%cwtgcell_old(begc:endc))
    allocate(this%dwt(begp:endp))
    allocate(this%growing_old_fraction(begp:endp))
    allocate(this%growing_new_fraction(begp:endp))

  end function constructor

  subroutine col_nf_init(this, begc, endc)
    !
    ! !ARGUMENTS:
    class(column_nitrogen_flux) :: this
    integer, intent(in) :: begc, endc

    allocate(this%hrv_deadstemn_to_prod10n        (begc:endc))                   ; this%hrv_deadstemn_to_prod10n       (:)   = spval
    allocate(this%hrv_deadstemn_to_prod100n       (begc:endc))                   ; this%hrv_deadstemn_to_prod100n      (:)   = spval
    allocate(this%m_n_to_litr_met_fire            (begc:endc,1:nlevdecomp_full)) ; this%m_n_to_litr_met_fire           (:,:) = spval
    allocate(this%m_n_to_litr_lig_fire            (begc:endc,1:nlevdecomp_full)) ; this%m_n_to_litr_lig_fire           (:,:) = spval
  end subroutine col_nf_init

    integer function shr_pio_getindex_fromid(compid) result(index)
     implicit none
     integer, intent(in) :: compid
     integer :: i

     index = -1
     do i=1,total_comps
        if(io_compid(i)==compid) then
          index = i
          exit
       end if
    end do

    if(index<0) then
       call shr_sys_abort('shr_pio_getindex :: compid out of allowed range')
    end if
  end function shr_pio_getindex_fromid


  integer function shr_pio_getindex_fromname(component) result(index)
     use shr_string_mod, only : shr_string_toupper

     implicit none

     ! 'component' must be equal to some element of io_compname(:)
     ! (but it is case-insensitive)
     character(len=*), intent(in) :: component

     character(len=len(component)) :: component_ucase
     integer :: i

     ! convert component name to upper case in order to match case in io_compname
     component_ucase = shr_string_toUpper(component)

     index = -1  ! flag for not found
     do i=1,size(io_compname)
        if (trim(component_ucase) == trim(io_compname(i))) then
           index = i
           exit
        end if
     end do
    if(index<0) then
       call shr_sys_abort(' shr_pio_getindex:: compid out of allowed range')
    end if
   end function shr_pio_getindex_fromname

  function get_gamma_A(ivt_in, elai240_in,elai_in,nclass_in)

    ! Activity factor for leaf age (Guenther et al., 2006)
    !-----------------------------
    ! If bgc not active, then elai is constant therefore gamma_a=1.0
    ! gamma_a set to unity for evergreens (Patches 1, 2, 4, 5)
    ! Note that we assume here that the time step is shorter than the number of 
    !days after budbreak required to induce isoprene emissions (ti=12 days) and 
    ! the number of days after budbreak to reach peak emission (tm=28 days)
    !
    ! !ARGUMENTS:
    implicit none
    integer,intent(in)  :: ivt_in
    integer,intent(in)  :: nclass_in
    real(r8),intent(in) :: elai240_in
    real(r8),intent(in) :: elai_in
    !
    ! !LOCAL VARIABLES:
    real(r8) :: get_gamma_A
    real(r8) :: elai_prev               ! lai for previous timestep
    real(r8) :: fnew, fgro, fmat, fold  ! fractions of leaves at different phenological stages
    !-----------------------------------------------------------------------
    !if ( (ivt_in == ndllf_dcd_brl_tree) .or. (ivt_in >= nbrdlf_dcd_trp_tree) ) then  ! non-evergreen
    if ( (woody(ivt_in) == 1.0_r8) .and. (evergreen(ivt_in) <= 0) ) then  ! non-evergreen tree
       
       if ( (elai240_in > 0.0_r8) .and. (elai240_in < 1.e30_r8) )then 
          elai_prev = 2._r8*elai240_in-elai_in  ! have accumulated average lai over last timestep
          if (elai_prev == elai_in) then
             fnew = 0.0_r8
             fgro = 0.0_r8
             fmat = 1.0_r8
             fold = 0.0_r8
          else if (elai_prev > elai_in) then
             fnew = 0.0_r8
             fgro = 0.0_r8
             fmat = 1.0_r8 - (elai_prev - elai_in)/elai_prev
             fold = (elai_prev - elai_in)/elai_prev
          else if (elai_prev < elai_in) then
             fnew = 1 - (elai_prev / elai_in)
             fgro = 0.0_r8
             fmat = (elai_prev / elai_in)
             fold = 0.0_r8
          end if
          
          get_gamma_A = fnew*Anew(nclass_in) + fgro*Agro(nclass_in) + fmat*Amat(nclass_in) + fold*Aold(nclass_in)

       else
          get_gamma_A = 1.0_r8
       end if
       
    else
       get_gamma_A = 1.0_r8
    end if
    
  end function get_gamma_A

   subroutine test_parsing_sub(bounds, var1, var2, input4, var3)

      type(bounds_type), intent(in) :: bounds
      real(r8), INTENT(IN) :: var1
      real(r8), INTENT(IN) :: var2(bounds%begg:)
      real(r8), INTENT(inout) :: var3
      logical  :: input4
      real(r8) :: x, y
      integer :: g
      g = 1

      if (input4)then 
         x = bounds%begg + var1 + var2(g)+ var3
         input4 = .not. input4
      else
         x = bounds%endg + var1 + var2(g) + var3
      end if

      call add(x, var3)

   end subroutine test_parsing_sub

   subroutine call_sub(numf, bounds, mytype, patch_state_updater)
      use shr_const_mod, only : test_type
      use constants_mod, only : i_const, param2

      integer, intent(in) :: numf
      type(bounds_type), intent(in) :: bounds
      type(test_type), INTENT(INOUT) :: mytype
      type(patch_state_updater_type), intent(inout) :: patch_state_updater

      real(r8) :: input1, input2(bounds%begg:bounds%endg), input3
      real(r8) :: local_var
      integer  :: g,j,c, N, i_type
      integer  :: lbj, ubj, jtop(1:numf)
      integer  :: soilc(1:numf)
      real(r8) :: a_tri(bounds%begc:bounds%endc,0:nlevdecomp+1)
      real(r8) :: b_tri(bounds%begc:bounds%endc,0:nlevdecomp+1)
      real(r8) :: c_tri(bounds%begc:bounds%endc,0:nlevdecomp+1)
      real(r8) :: r_tri(bounds%begc:bounds%endc,0:nlevdecomp+1)
      real(r8) :: u_tri(bounds%begc:bounds%endc,0:nlevdecomp+1)
      real(r8), pointer :: test_ptr(:,:)
      integer, parameter :: four = 4

      associate( &
        hrv_deadstemn_to_prod10n  => col_nf%hrv_deadstemn_to_prod10n, &
        psu => patch_state_updater, &
        field1 => mytype%field1 &
      )

      i_type = i_const - 3

      psu%dwt(:) = 19.8
      if (use_fates) then 
         do i =1, elm_fates%num
            elm_fates%field(i) = field1(i)
         end do
        ft_index_bigleaf(c) = field1(c)
      elseif(c == 4) then 
         if (i_type == four) then 
            soilc(1) = four
         end if 
      end if

      select case (i_type)
      case (1)  ! C
         test_ptr => col_nf%m_n_to_litr_met_fire
      case (2)  ! N
         test_ptr => col_nf%m_n_to_litr_lig_fire
      end select

      filter(i_type)%num_soilc = 10
      filter(i_type)%soilc(:) = 4

      mytype%field2(:) = test_ptr(:,1)*param2(:)

      field1(c) = shr_const_pi*mytype%field2(c)

      call test_parsing_sub(bounds, max(input1*shr_const_pi, local_var+input2(g)), &
         col_nf%m_n_to_litr_met_fire(c,1:N), landunit_is_special(g),var3=input3)

      call Tridiagonal(bounds, -lbj+1, four, jtop, numf, soilc, a_tri, b_tri, c_tri, r_tri, u_tri)
      call col_nf%Init(bounds%begc, bounds%endc)

      call ptr_test_sub(filter(i_type)%num_soilc, filter(i_type)%soilc, test_ptr)

      call trace_dtype_example(mytype, col_nf, .true.)
      veg_rootc_bigleaf(:,:) = 2

      test_ptr(:,:) = SHR_CONST_SPVAL

      end associate

      if(.true.) field1(c) = SHR_CONST_PI
   end subroutine call_sub

   subroutine trace_dtype_example(mytype2, col_nf_inst, flag)
      type(test_type) , INTENT(INOUT) :: mytype2
      type(column_nitrogen_flux), INTENT(INOUT) :: col_nf_inst
      logical, intent(in) :: flag
      integer :: i
      associate(&
          field1 => mytype2%field1,&
          field2 => mytype2%field2,&
          !

          field3 => mytype2%field3,&
          field4 => mytype2%field4,&
          hrv => col_nf_inst%hrv_deadstemn_to_prod10n &
      )
      if ( mytype2%active .or. flag )then
         do i=1,10
            field2(i) = field2(i)/field1(i) + field3(i)
            hrv(i) = field2(i)
            call add(field2(i), field4(i))
         end do
      end if
      end associate
   end subroutine trace_dtype_example

   subroutine ptr_test_sub(numf, soilc, arr)
      integer , intent(in) :: numf
      integer, intent(in) :: soilc(:)
      real(r8) , intent(inout) :: arr(:,:)

      arr(:,:) = 1.0_r8
   end subroutine ptr_test_sub

   function landunit_is_special(ltype) result(is_special)
      !
      ! !DESCRIPTION:
      ! Returns true if the landunit type ltype is a special landunit; returns false otherwise
      !
      ! !USES:
      !
      ! !ARGUMENTS:
      logical :: is_special  ! function result
      integer :: ltype       ! landunit type of interest
      !
      ! !LOCAL VARIABLES:

      character(len=*), parameter :: subname = 'landunit_is_special'
      !-----------------------------------------------------------------------

      SHR_ASSERT((ltype >= 1 .and. ltype <= max_lunit), subname//': ltype out of bounds')

      if (ltype == istsoil .or. ltype == istcrop) then
         is_special = .false.
      else
         is_special = .true.
      end if

   end function landunit_is_special

   type(prior_weights_type) function weight_constructor(bounds)
      !
      ! !DESCRIPTION:
      ! Initialize a prior_weights_type object
      !
      ! !ARGUMENTS:
      type(bounds_type), intent(in) :: bounds   ! processor bounds
      !
      ! !LOCAL VARIABLES:

      character(len=*), parameter :: subname = 'prior_weights_type constructor'
      ! ----------------------------------------------------------------------


      allocate (weight_constructor%pwtcol(bounds%begp:bounds%endp))
      allocate (weight_constructor%cactive(bounds%begc:bounds%endc))

   end function weight_constructor

   function old_weight_was_zero(this, bounds)
      !
      ! !DESCRIPTION:
      ! Returns a patch-level logical array that is true wherever the patch weight was zero
      ! prior to weight updates
      !
      ! !USES:
      !
      ! !ARGUMENTS:
      class(patch_state_updater_type), intent(in) :: this
      type(bounds_type), intent(in) :: bounds
      logical :: old_weight_was_zero(bounds%begp:bounds%endp)  ! function result
      !
      ! !LOCAL VARIABLES:
      integer :: p

      character(len=*), parameter :: subname = 'old_weight_was_zero'
      !-----------------------------------------------------------------------

      do p = bounds%begp, bounds%endp
         old_weight_was_zero(p) = (this%pwtgcell_old(p) == 0._r8)
      end do

   end function old_weight_was_zero

   logical function CNAllocate_carbon_only()
      cnallocate_carbon_only = .true.
   end function cnallocate_carbon_only

   pure function get_beg(bounds, subgrid_level) result(beg_index)
      !
      ! !DESCRIPTION:
      ! Get beginning bounds for a given subgrid level
      !
      ! subgrid_level should be one of the constants defined in this module:
      ! BOUNDS_SUBGRID_GRIDCELL, BOUNDS_SUBGRID_LANDUNIT, etc.
      !
      ! Returns -1 for invalid subgrid_level (does not abort in this case, in order to keep
      ! this function pure).
      !
      ! !USES:
      !
      ! !ARGUMENTS:
      integer :: beg_index  ! function result
      type(bounds_type), intent(in) :: bounds
      integer, intent(in) :: subgrid_level
      !
      ! !LOCAL VARIABLES:

      character(len=*), parameter :: subname = 'get_beg'
      !-----------------------------------------------------------------------

      select case (subgrid_level)
      case (BOUNDS_SUBGRID_GRIDCELL)
         beg_index = bounds%begg
      case (BOUNDS_SUBGRID_TOPOUNIT)
         beg_index = bounds%begt
      case (BOUNDS_SUBGRID_LANDUNIT)
         beg_index = bounds%begl
      case (BOUNDS_SUBGRID_COLUMN)
         beg_index = bounds%begc
      case (BOUNDS_SUBGRID_PATCH)
         beg_index = bounds%begp
      case (BOUNDS_SUBGRID_COHORT)
         beg_index = bounds%begCohort
      case default
         beg_index = -1
      end select

   end function get_beg

  !-----------------------------------------------------------------------
  subroutine Tridiagonal_sr (bounds, lbj, ubj, jtop, numf, filter, a, b, c, r, u, is_col_active)
    !$acc routine seq
    ! !DESCRIPTION:
    ! Tridiagonal matrix solution
    ! A x = r
    ! where x and r are vectors
    ! !USES:
    !
    ! !ARGUMENTS:
    implicit none
    type(bounds_type) , intent(in)    :: bounds                                   ! bounds
    integer           , intent(in)    :: lbj, ubj                                 ! lbinning and ubing level indices
    integer           , intent(in)    :: jtop( bounds%begc: bounds%endc)          ! top level for each column [col]
    integer           , intent(in)    :: numf                                     ! filter dimension
    integer           , intent(in)    :: filter(:)                                ! filter
    real(r8)          , intent(in)    :: a( bounds%begc:bounds%endc , lbj:ubj)    ! "a" left off diagonal of tridiagonal matrix [col , j]
    real(r8)          , intent(in)    :: b( bounds%begc:bounds%endc , lbj:ubj)    ! "b" diagonal column for tridiagonal matrix [col  , j]
    real(r8)          , intent(in)    :: c( bounds%begc:bounds%endc , lbj:ubj)    ! "c" right off diagonal tridiagonal matrix [col   , j]
    real(r8)          , intent(in)    :: r( bounds%begc:bounds%endc , lbj:ubj)    ! "r" forcing term of tridiagonal matrix [col      , j]
    real(r8)          , intent(inout) :: u( bounds%begc:bounds%endc , lbj:ubj)    ! solution [col                                    , j]
                                                                                  !
    integer                           :: j,ci,fc                                  ! indices
    logical, optional, intent(in)     :: is_col_active(bounds%begc:bounds%endc)   !
    logical                           :: l_is_col_active(bounds%begc:bounds%endc) !
    real(r8)                          :: gam(bounds%begc:bounds%endc,lbj:ubj)     ! temporary
    real(r8)                          :: bet(bounds%begc:bounds%endc)             ! temporary

    character(len=255)                :: subname ='Tridiagonal_sr'
    !-----------------------------------------------------------------------

    ! Solve the matrix
    if(present(is_col_active))then
       l_is_col_active(:) = is_col_active(:)
    else
       l_is_col_active(:) = .true.
    endif

    do fc = 1,numf
        ci = filter(fc)
        if(l_is_col_active(ci))then
            bet(ci) = b(ci,jtop(ci))
        endif
    end do

    do j = lbj, ubj
       do fc = 1,numf
           ci = filter(fc)
           if(l_is_col_active(ci))then
             if (j >= jtop(ci)) then
               if (j == jtop(ci)) then
                 u(ci,j) = r(ci,j) / bet(ci)
               else
                 gam(ci,j) = c(ci,j-1) / bet(ci)
                 bet(ci) = b(ci,j) - a(ci,j) * gam(ci,j)
                 u(ci,j) = (r(ci,j) - a(ci,j)*u(ci,j-1)) / bet(ci)
               end if
             end if
           endif
        end do
    end do

    do j = ubj-1,lbj,-1
        do fc = 1,numf
           ci = filter(fc)
           if(l_is_col_active(ci))then
             if (j >= jtop(ci)) then
               u(ci,j) = u(ci,j) - gam(ci,j+1) * u(ci,j+1)
             end if
           endif
        end do
    end do


  end subroutine Tridiagonal_sr

  !-----------------------------------------------------------------------
  subroutine Tridiagonal_mr (bounds, lbj, ubj, jtop, numf, filter, ntrcs, a, b, c, r, u, is_col_active)
    !$acc routine seq
    ! !DESCRIPTION:
    ! Tridiagonal matrix solution
    ! A X = R
    ! where A, X and R are all matrices.
    ! !USES:
    !
    ! !ARGUMENTS:
    implicit none
    type(bounds_type) , intent(in)    :: bounds                                         ! bounds
    integer           , intent(in)    :: lbj, ubj                                       ! lbinning and ubing level indices
    integer           , intent(in)    :: jtop( bounds%begc: bounds%endc)                ! top level for each column [col]
    integer           , intent(in)    :: numf                                           ! filter dimension
    integer           , intent(in)    :: ntrcs                                          !
    integer           , intent(in)    :: filter(:)                                      ! filter
    real(r8)          , intent(in)    :: a( bounds%begc:bounds%endc , lbj:ubj)          ! "a" left off diagonal of tridiagonal matrix [col , j]
    real(r8)          , intent(in)    :: b( bounds%begc:bounds%endc , lbj:ubj)          ! "b" diagonal column for tridiagonal matrix [col  , j]
    real(r8)          , intent(in)    :: c( bounds%begc:bounds%endc , lbj:ubj)          ! "c" right off diagonal tridiagonal matrix [col   , j]
    real(r8)          , intent(in)    :: r( bounds%begc:bounds%endc , lbj:ubj, 1:ntrcs) ! "r" forcing term of tridiagonal matrix [col , j]
    real(r8)          , intent(inout) :: u( bounds%begc:bounds%endc , lbj:ubj, 1:ntrcs) ! solution [col, j]
                                                                                        !
    integer                           :: j,ci,fc,k                                      ! indices
    logical, optional, intent(in)     :: is_col_active(bounds%begc:bounds%endc)         !
    logical                           :: l_is_col_active(bounds%begc:bounds%endc)       !
    real(r8)                          :: gam(bounds%begc:bounds%endc,lbj:ubj)           ! temporary
    real(r8)                          :: bet(bounds%begc:bounds%endc)                   ! temporary

    character(len=255) :: subname ='Tridiagonal_sr'
    !-----------------------------------------------------------------------

    ! Solve the matrix
    if (present(is_col_active)) then
       l_is_col_active(:) = is_col_active(:)
    else
       l_is_col_active(:) = .true.
    endif

    do fc = 1,numf
       ci = filter(fc)
       if (l_is_col_active(ci))then
          bet(ci) = b(ci,jtop(ci))
       endif
    end do

    do j = lbj, ubj
       do fc = 1,numf
          ci = filter(fc)
          if (l_is_col_active(ci))then
             if (j >= jtop(ci)) then
                if (j == jtop(ci))then
                   do k = 1, ntrcs
                     u(ci,j,k) = r(ci,j,k)/bet(ci)
                   enddo
                else
                   gam(ci,j) = c(ci,j-1) / bet(ci)
                   bet(ci) = b(ci,j) - a(ci,j) * gam(ci,j)
                   do k = 1, ntrcs
                      u(ci,j,k) = (r(ci,j, k) - a(ci,j)*u(ci,j-1, k)) / bet(ci)
                    end do
                 end if
             end if
          end if
       end do
    end do


    do j = ubj-1,lbj,-1
       do fc = 1,numf
          ci = filter(fc)
          if (l_is_col_active(ci)) then
             if (j >= jtop(ci)) then
                do k = 1, ntrcs
                  u(ci,j, k) = u(ci,j, k) - gam(ci,j+1) * u(ci,j+1, k)
                end do
             end if
          end if
       end do
    end do


  end subroutine Tridiagonal_mr

  subroutine add(x,y)
   real(r8), intent(in) :: x
   real(r8), intent(inout) :: y

   y = x+y

   end subroutine add

end module test_sub_parse
