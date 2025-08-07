from scripts.fortran_parser.lexer import Lexer
from scripts.fortran_parser.spel_parser import Parser
from scripts.fortran_parser.tracing import Trace
from scripts.types import LineTuple, LogicalLineIterator

var_txt = """
    ! 1) Old-style, no attributes (no :: allowed/used)
    integer i, j, k

    real   (SHR_KIND_R8), parameter :: obamp(poblen) =  & ! amplitudes for obliquity cos series
         &      (/   -2462.2214466_SHR_KIND_R8, -857.3232075_SHR_KIND_R8, -629.3231835_SHR_KIND_R8,   &
         &            -414.2804924_SHR_KIND_R8, -311.7632587_SHR_KIND_R8,  308.9408604_SHR_KIND_R8,   &
         &            -162.5533601_SHR_KIND_R8, -116.1077911_SHR_KIND_R8,  101.1189923_SHR_KIND_R8,   &
         &             -67.6856209_SHR_KIND_R8,   24.9079067_SHR_KIND_R8,   22.5811241_SHR_KIND_R8,   &
         &             -21.1648355_SHR_KIND_R8,  -15.6549876_SHR_KIND_R8,   15.3936813_SHR_KIND_R8,   &
         &              14.6660938_SHR_KIND_R8,  -11.7273029_SHR_KIND_R8,   10.2742696_SHR_KIND_R8,   &
         &               6.4914588_SHR_KIND_R8,    5.8539148_SHR_KIND_R8,   -5.4872205_SHR_KIND_R8,   &
         &              -5.4290191_SHR_KIND_R8,    5.1609570_SHR_KIND_R8,    5.0786314_SHR_KIND_R8,   &
         &              -4.0735782_SHR_KIND_R8,    3.7227167_SHR_KIND_R8,    3.3971932_SHR_KIND_R8,   &
         &              -2.8347004_SHR_KIND_R8,   -2.6550721_SHR_KIND_R8,   -2.5717867_SHR_KIND_R8,   &
         &              -2.4712188_SHR_KIND_R8,    2.4625410_SHR_KIND_R8,    2.2464112_SHR_KIND_R8,   &
         &              -2.0755511_SHR_KIND_R8,   -1.9713669_SHR_KIND_R8,   -1.8813061_SHR_KIND_R8,   &
         &              -1.8468785_SHR_KIND_R8,    1.8186742_SHR_KIND_R8,    1.7601888_SHR_KIND_R8,   &
         &              -1.5428851_SHR_KIND_R8,    1.4738838_SHR_KIND_R8,   -1.4593669_SHR_KIND_R8,   &
         &               1.4192259_SHR_KIND_R8,   -1.1818980_SHR_KIND_R8,    1.1756474_SHR_KIND_R8,   &
         &              -1.1316126_SHR_KIND_R8,    1.0896928_SHR_KIND_R8/)


    ! 2) Attributes present → must use ::
    integer*8, intent(in), value :: n_in

    ! 3) Kind by named constant (via iso_fortran_env)
    integer(int64), parameter :: big = 9223372036854775807_int64
    real(real64), save :: r64

    type, extends(soil_water_retention_curve_type) :: &
         soil_water_retention_curve_clapp_hornberg_1978_type
       private
     contains
       procedure :: soil_hk              ! compute hydraulic conductivity
       procedure :: soil_suction         ! compute soil suction potential
       procedure :: soil_suction_inverse ! compute relative saturation at which soil suction is equal to a target value
    end type soil_water_retention_curve_clapp_hornberg_1978_type

    character(len=32),public :: type
    type(type) :: var

     type, abstract, extends(fire_method_type) :: fire_base_type
       private
       !PRIVATE MEMBER DATA:

         real(r8), public, pointer :: forc_lnfm(:) => null()    ! Lightning frequency
         real(r8), public, pointer :: forc_hdm(:)     ! Human population density

         real(r8), public, pointer :: gdp_lf_col(:)   ! col global real gdp data (k US$/capita)

         type(shr_strdata_type) :: sdat_hdm           ! Human population density input data stream
         type(shr_strdata_type) :: sdat_lnfm          ! Lightning input data stream
       contains

         ! !PUBLIC MEMBER FUNCTIONS:
         procedure, public :: FireInit => BaseFireInit                           ! Initialization of Fire
         procedure, public :: BaseFireInit                                       ! Initialization of Fire
         procedure, public :: FireInterp                                         ! Interpolate fire data
         procedure(need_lightning_and_popdens_interface), public, deferred :: &
              need_lightning_and_popdens ! Returns true if need lightning & popdens

         ! !PRIVATE MEMBER FUNCTIONS:
         procedure, private :: hdm_init     ! position datasets for dynamic human population density
         procedure, private :: hdm_interp   ! interpolates between two years of human pop. density file data
         procedure, private :: lnfm_init    ! position datasets for Lightning
         procedure, private :: lnfm_interp  ! interpolates between two years of Lightning file data

     end type fire_base_type

      type domain_type
         integer          :: ns         ! global size of domain
         integer          :: ni,nj      ! global axis if 2d (nj=1 if unstructured)
         logical          :: isgrid2d   ! true => global grid is lat/lon
         integer          :: nbeg,nend  ! local beg/end indices
         character(len=32):: elmlevel   ! grid type
         integer ,pointer :: mask(:)    ! land mask: 1 = land, 0 = ocean
         real(r8),pointer :: frac(:)    ! fractional land
                                        ! 0=SMB not required (default)
                                        ! (glcmask is just a guess at the appropriate mask, known at initialization - in contrast to icemask, which is the true mask obtained from glc)
         logical          :: set        ! flag to check if domain is set
         logical          :: decomped   ! decomposed locally or global copy
     
      end type domain_type

    ! 4) Character with explicit LEN and assumed-len dummy
    character(len=10) :: tag
    character(*), intent(in) :: arg

    ! 5) Deferred-length character, allocatable
    character(len=:), allocatable :: dyn_name

    ! 6) Array with explicit bounds and separate DIMENSION attr
    real, dimension(0:n-1, m) :: A, B

    ! 7) Assumed-shape dummy with INTENT
    real, intent(in) :: x(:), y(:, :)

    ! 8) Deferred-shape allocatable array with TARGET (and CONTIGUOUS)
    real, allocatable, target, contiguous :: buf(:)

    ! 9) Pointer with default NULL() initialization
    real, pointer :: pvec(:) => null()

    ! 10) Pointer/target pairing with association
    integer, target :: t
    integer, pointer :: tp => t

    ! 12) Complex with kind inferred from literal
    complex(kind=kind(1.0d0)) :: z

    ! 13) LOGICAL with VOLATILE and array constructor init
    logical, volatile :: flags(0:n-1) = [ i == 0, i=0, n-1 ]

    ! 14) PARAMETER with expressions and array constructors (/ /) and [ ]
    integer, parameter :: m = 2*n_in + 3
    integer, parameter :: idxs(4) = (/ 1, 3, 5, 7 /)
    real,    parameter :: w(:) = [ 0.1_real64, 0.2_real64, 0.7_real64 ]


    ! 18) BIND(C) scalars using iso_c_binding
    integer(c_int), bind(C, name="n_items") :: n_items
    real(c_double), bind(C) :: x_c
    type(c_ptr) :: raw_handle
    character(kind=c_char), dimension(0:15), bind(C, name="tag") :: ctag

    ! 19) DIMENSION attribute on the statement, entity-specific bounds on names
    real, dimension(:), allocatable :: u(:), v(0:n-1)

    ! 20) Mixed entities with different array specs and initializations
    integer :: counts(0:n-1) = 0, total = 0
    real    :: grid(1:n, 1:m) = 0.0_real64

    ! 22) VOLATILE and VALUE (VALUE only valid for dummy args)
    integer, value :: byval_arg

    ! 25) A trickier CHARACTER with kind and len expressions
    character(len=2*n_in+1, kind=selected_char_kind('DEFAULT')) :: label

    ! 27) PARAMETER using RESHAPE in initialization
    real, parameter :: W(2,3) = reshape([1.0,2.0,3.0,4.0,5.0,6.0], [2,3], pad=[0.0])

    real(r8) :: w(2)

  type, public :: spm_list_type
    real(r8) :: val
    integer  :: icol
    integer  :: irow
    type(spm_list_type), pointer :: next => null()
  end type spm_list_type

  type :: lom_type
  contains
    procedure :: calc_state_pscal
    procedure :: calc_reaction_rscal
    procedure :: apply_reaction_rscal
  end type lom_type
  type, public :: sparseMat_type
    real(r8), pointer :: val(:)      !nonzero val
    integer , pointer :: icol(:)     !col id for each val
    integer , pointer :: pB(:)       !first nonzero column id (as in val) in each row
    integer , pointer :: ncol(:)     !number of nonzero columns in each row
    integer :: szcol                 !number of columns of the matrix
    integer :: szrow
  contains
    procedure, public :: init
  end type sparseMat_type
  real(r8), parameter :: tiny_val=1.e-14_r8

  class(sparseMat_type), pointer :: spm_carbon_p,spm_carbon_d
  class(sparseMat_type), pointer :: spm_nutrient_p,spm_nutrient_d
  type(spm_list_type), pointer :: spm_list
    """

ifs_txt = """
   if (use_fates) then
      elmtype_to_not_not_add = 2
      if (nu_com .eq. 'RD') then
          blah=3
      elseif(nu_com .eq. 'ECA') then

        do j = 1, n
            x(fc) = x(fc) + b(nc,j)*dz(j)
            print *, "j :" , j
        end do
         elmtype2 = .4_r8
           if(use_vertsoilc) then
              y = 10
           end if
      else
         elm_type=2
      end if 
      elmtype_to_add = 1
        do j = 1, n
            x(fc) = x(fc) + b(nc,j)*dz(j)
            print *, "j :" , j
        end do
   else 
       if ( 1> 2 .and. x <= y) then
          elmtype2 = z
       end if
   end if
   if(z > 5e-3 .and. z < 8d-3) elm_type = 3
   if ( elmvarctl_isset )then
      call shr_sys_abort(' ERROR:: control variables already set, cannot call this routine')
   end if
   if (forest_fert_exp) then
        if ( ((fert_continue(c) == 1 .and. kyr > fert_start(c) .and. kyr <= fert_end(c)) .or.  kyr == fert_start(c)) &
             .and. fert_type(c) == 1 &
             .and. kda == 1  .and. mcsec == 1800) then ! fertilization assumed to occur at the begnining of each month
           col_ninputs(c) = col_ninputs(c) + fert_dose(c,kmo)/dt
        end if
     end if

   if ( present(single_column_in) ) single_column = single_column_in
   if (use_vertsoilc) then
    ptr2d => this%fpi_vr_col
    ptr2d => this%fpi_p_vr_col
   else
    ptr1d => this%fpi_vr_col(:,1)
    ptr1d => this%fpi_p_vr_col(:,1)
   end if
    #define YES
    #ifdef YES

    help = x*y
    #endif
    if (flag=='read' .and. .not. readvar) then
        write(iulog) 'initializing this%ctrunc with atmospheric c14 value'
        write(iulog,'(IA4.)') help, sh(dasdf)
        do i = bounds%begp,bounds%endp
        end do
    end if
            do while( (abs((ustar - ustar_prev)/ustar) > flux_con_tol .or. &
             abs(tau_diff) > dtaumin) .and. &
             iter < flux_con_max_iter)
           iter = iter + 1_in
           ustar_prev = ustar
           zetu=cc*ribu/( 1.0_r8 - (.004_r8*beta**3*zi/zu) * ribu )
           if (present(wsresp) .and. present(tau_est)) then
              ! Update stress and magnitude of mean wind.
              call shr_flux_update_stress(wind0, wsresp(n), tau_est(n), &
                   tau, prev_tau, tau_diff, prev_tau_diff, wind_adj)
              vmag = wind_adj
           end if
        enddo

   if (use_lch4) then
      if (organic_max > 0._r8) then
         om_frac = min(cellorg(c,j)/organic_max, 1._r8)
         ! Use first power, not square as in iniTimeConst
      else
         om_frac = 1._r8
      end if
      ratio_diffusivity_water_gas(c,j) = (d_con_g(2,1) + d_con_g(2,2)*t_soisno(c,j) ) * 1.e-4_r8 / &
           ((d_con_w(2,1) + d_con_w(2,2)*t_soisno(c,j) + d_con_w(2,3)*t_soisno(c,j)**2) * 1.e-9_r8)

      if (o2_decomp_depth_unsat(c,j) /= spval .and. conc_o2_unsat(c,j) /= spval .and.  &
           o2_decomp_depth_unsat(c,j) > smallparameter) then
         anaerobic_frac(c,j) = exp(-rij_kro_a * r_psi(c,j)**(-rij_kro_alpha) * &
              o2_decomp_depth_unsat(c,j)**(-rij_kro_beta) * &
              conc_o2_unsat(c,j)**rij_kro_gamma * (h2osoi_vol(c,j) + ratio_diffusivity_water_gas(c,j) * &
              watsat(c,j))**rij_kro_delta)
      else
         anaerobic_frac(c,j) = 0._r8
      endif
    end if 
   """


def test_if_parsing():
    parse_statements(ifs_txt)
    return


def test_do_parsing():
    txt = """
    do nc = 1, nclumps, 2
        do fc = 1, filter(nc)%num_soilc
            x(fc) = y(nc)
            y(nc) = z%huh(fc)
        end do

        soilc(:) = 1

        do j = 1, n
            x(fc) = x(fc) + b(nc,j)*dz(j)
            print *, "j :" , j
        end do
    end do

    do

       iter=iter+1

       call spacF(p,c,x,f,qflx_sun,qflx_sha, &
            atm2lnd_inst,canopystate_inst,soilstate_inst )

       if ( sqrt(sum(f*f)) < tolf*(qflx_sun+qflx_sha) ) then  !fluxes balanced -> exit
          flag = .false.
          exit
       end if
       if ( iter>itmax ) then                                 !exceeds max iters -> exit
          flag = .false.
          exit
       end if
    end do
    """
    lines = [LineTuple(line=l, ln=i) for i, l in enumerate(txt.split("\n"))]

    line_it = LogicalLineIterator(lines=lines, log_name="test_dos")
    lexer = Lexer(line_it=line_it)
    parser = Parser(lex=lexer)
    program = parser.parse_program()

    for stmt in program.statements:
        print(stmt)


def test_var_parsing():
    Trace.enabled = True
    parse_statements(var_txt)


def parse_statements(txt: str):
    lines = [LineTuple(line=l, ln=i) for i, l in enumerate(txt.split("\n"))]

    # Trace.enabled = True
    line_it = LogicalLineIterator(lines=lines, log_name="test_ifs")
    for fl, x in line_it:
        print(x, fl)
    line_it.reset()
    lexer = Lexer(line_it=line_it)
    parser = Parser(lex=lexer)
    program = parser.parse_program()

    parser.logger.info(f"FOUND {len(program.statements)} statements")

    for stmt in program.statements:
        print(stmt)
