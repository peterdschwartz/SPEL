module decompInitMod
#include "unittest_defs.h"

  !------------------------------------------------------------------------------
  ! !DESCRIPTION:
  ! Module provides a descomposition into a clumped data structure which can
  ! be mapped back to atmosphere physics chunks.
  !
  ! !USES:
  use shr_kind_mod    , only : r8 => shr_kind_r8
  !use shr_sys_mod     , only : shr_sys_flush
  !use shr_log_mod     , only : errMsg => shr_log_errMsg
  !use spmdMod         , only : masterproc, iam, npes, mpicom, comp_id
  !use abortutils      , only : endrun
  use elm_varctl      , only : iulog, use_fates
  use elm_varcon      , only : grlnd
  use GridcellType    , only : grc_pp
  use LandunitType    , only : lun_pp                
  use TopounitType    , only : top_pp                
  use ColumnType      , only : col_pp                
  !use FatesInterfaceTypesMod, only : fates_maxElementsPerSite
  use VegetationType  , only : veg_pp                
  use decompMod
  !use mct_mod  
  use topounit_varcon   , only : max_topounits, has_topounit
  use domainMod         , only: ldomain
  !
  ! !PUBLIC TYPES:
  implicit none
  !
  ! !PUBLIC MEMBER FUNCTIONS:
  public decompInit_clumps       ! initializes atm grid decomposition into clumps
  public decompInit_lnd_simple   ! initializes lnd grid decomposition into clumps and processors using simple approach of ngrids/nclumps
  !
  ! !PRIVATE TYPES:
  private
  integer, pointer :: lcid(:)       ! temporary for setting ldecomp
  !------------------------------------------------------------------------------

contains


  !------------------------------------------------------------------------------
  subroutine decompInit_lnd_simple(lni,lnj,amask)
    !
    ! !DESCRIPTION:
    ! This subroutine initializes the land surface decomposition into a clump
    ! data structure using a "simple" approach of dividing the number of active
    ! land grid cells by number of clumps. This assumes each pe has the same number of clumps
    ! set by clump_pproc.
    !
    ! !ARGUMENTS:
    implicit none
    integer , intent(in) :: amask(:)
    integer , intent(in) :: lni,lnj   ! domain global size
    !
    ! !LOCAL VARIABLES:
    integer              :: lns                    ! global domain size
    integer              :: ln,lj                  ! indices
    integer              :: ag,an,ai,aj            ! indices
    integer              :: numg                   ! number of land gridcells
    logical              :: seglen1                ! is segment length one
    real(r8)             :: seglen                 ! average segment length
    real(r8)             :: rcid                   ! real value of cid
    integer              :: cid,pid                ! indices
    integer              :: n,m,ng                 ! indices
    integer              :: ier                    ! error code
    integer              :: beg,end,lsize,gsize    ! used for gsmap init
    integer, pointer     :: gindex(:)              ! global index for gsmap init
    integer, pointer     :: clumpcnt(:)            ! clump index counter
    integer, allocatable :: proc_ncell(:)          ! number of cells assigned to a process
    integer, allocatable :: proc_begg(:)           ! beginning cell index assigned to a process
    integer              :: numg_per_clumps        ! number of land grids per clump ( = numg/nclump)
    integer              :: numg_per_clumps_mod    ! modulo of numg/nclump
    integer              :: cur_cid                ! current clump id
    integer              :: numg_for_cur_clump     ! a local counter for the number of land grid for the current clump
    integer              :: max_numg_for_cur_clump ! maximum number of land grids for the current clump
    !------------------------------------------------------------------------------

    lns = lni * lnj

    !--- set and verify nclumps ---
    if (clump_pproc > 0) then
       nclumps = clump_pproc * npes
       if (nclumps < npes) then
          write(iulog,*) 'decompInit_lnd(): Number of gridcell clumps= ',nclumps, &
               ' is less than the number of processes = ', npes
          !! call_endrun(msg=errMsg(__FILE__, __LINE__))
       end if
    else
       write(iulog,*)'clump_pproc= ',clump_pproc,'  must be greater than 0'
       !! call_endrun(msg=errMsg(__FILE__, __LINE__))
    end if

    ! allocate and initialize procinfo (from decompMod.F90) and clumps 
    ! beg and end indices initialized for simple addition of cells later 

    allocate(procinfo%cid(clump_pproc), stat=ier)
    if (ier /= 0) then
       write(iulog,*) 'decompInit_lnd(): allocation error for procinfo%cid'
       !! call_endrun(msg=errMsg(__FILE__, __LINE__))
    endif
    procinfo%nclumps = clump_pproc
    procinfo%cid(:)  = -1
    procinfo%ncells  = 0
    procinfo%ntunits  = 0
    procinfo%nlunits = 0
    procinfo%ncols   = 0
    procinfo%npfts   = 0
    procinfo%nCohorts = 0
    procinfo%begg    = 1
    procinfo%begt    = 1
    procinfo%begl    = 1
    procinfo%begc    = 1
    procinfo%begp    = 1
    procinfo%begCohort    = 1
    procinfo%endg    = 0
    procinfo%endt    = 0
    procinfo%endl    = 0
    procinfo%endc    = 0
    procinfo%endp    = 0
    procinfo%endCohort    = 0

    allocate(clumps(nclumps), stat=ier)
    if (ier /= 0) then
       write(iulog,*) 'decompInit_lnd(): allocation error for clumps'
       ! call_endrun(msg=errMsg(__FILE__, __LINE__))
    end if
    clumps(:)%owner   = -1
    clumps(:)%ncells  = 0
    clumps(:)%ntunits = 0
    clumps(:)%nlunits = 0
    clumps(:)%ncols   = 0
    clumps(:)%npfts   = 0
    clumps(:)%nCohorts = 0
    clumps(:)%begg    = 1
    clumps(:)%begt    = 1
    clumps(:)%begl    = 1
    clumps(:)%begc    = 1
    clumps(:)%begp    = 1
    clumps(:)%begCohort    = 1
    clumps(:)%endg    = 0
    clumps(:)%endt    = 0
    clumps(:)%endl    = 0
    clumps(:)%endc    = 0
    clumps(:)%endp    = 0
    clumps(:)%endCohort    = 0

    ! assign clumps to proc round robin 
    cid = 0
    do n = 1,nclumps
       pid = mod(n-1,npes)
       if (pid < 0 .or. pid > npes-1) then
          write(iulog,*) 'decompInit_lnd(): round robin pid error ',n,pid,npes
          ! call_endrun(msg=errMsg(__FILE__, __LINE__))
       endif
       clumps(n)%owner = pid
       if (iam == pid) then
          cid = cid + 1
          if (cid < 1 .or. cid > clump_pproc) then
             write(iulog,*) 'decompInit_lnd(): round robin pid error ',n,pid,npes
             ! call_endrun(msg=errMsg(__FILE__, __LINE__))
          endif
          procinfo%cid(cid) = n
       endif
    enddo

    ! count total land gridcells
    numg = 0
    do ln = 1,lns
       if (amask(ln) == 1) then
          numg = numg + 1
       endif
    enddo
   
    if (npes > numg) then
       write(iulog,*) 'decompInit_lnd(): Number of processes exceeds number ', &
            'of land grid cells',npes,numg
       !! call_endrun(msg=errMsg(__FILE__, __LINE__))
    end if
    if (nclumps > numg) then
       write(iulog,*) 'decompInit_lnd(): Number of clumps exceeds number ', &
            'of land grid cells',nclumps,numg
       !! call_endrun(msg=errMsg(__FILE__, __LINE__))
    end if

    numg_per_clumps = int(dble(numg)/dble(nclumps))
    numg_per_clumps_mod = numg - int(dble(numg)/dble(nclumps)) * nclumps

    ! Assign gridcells to clumps (and thus pes) ---

    allocate(lcid(lns), stat=ier)
    if (ier /= 0) then
       write(iulog,*) 'decompInit_lnd(): allocation error for lcid'
       ! call_endrun(msg=errMsg(__FILE__, __LINE__))
    end if

    lcid(:) = 0
    ng = 0    
    cur_cid = 1
    numg_for_cur_clump = 0
    max_numg_for_cur_clump = numg_per_clumps + numg_per_clumps_mod

    do ln = 1,lns
       if (amask(ln) == 1) then
          ng = ng  + 1

          ! increment the number of land grid cells for the current clump
          numg_for_cur_clump = numg_for_cur_clump + 1

          ! check if the number of land grid cells has exceeded the max number of
          ! grid cells for the current clump
          if (numg_for_cur_clump > max_numg_for_cur_clump) then

            ! increment the clump id
            cur_cid = cur_cid + 1

            ! determine the max number of grid cell for the new clump
            if (cur_cid <= numg_per_clumps_mod) then
                max_numg_for_cur_clump = numg_per_clumps + 1
             else
                max_numg_for_cur_clump = numg_per_clumps
             end if

             ! reset the number of land grid cells for the current clump
             numg_for_cur_clump = 0
          end if
          cid = cur_cid
          lcid(ln) = cid

          !--- give gridcell cell to pe that owns cid ---
          !--- this needs to be done to subsequently use function
          !--- get_proc_bounds(begg,endg) 
          if (iam == clumps(cid)%owner) then
             procinfo%ncells  = procinfo%ncells  + 1
          endif

          !--- give gridcell to cid ---
          clumps(cid)%ncells  = clumps(cid)%ncells  + 1

       end if
    enddo

    ! calculate number of cells per process
    allocate(proc_ncell(0:npes-1), stat=ier)
    if (ier /= 0) then
       write(iulog,*) 'decompInit_lnd(): allocation error for proc_ncell'
       ! call_endrun(msg=errMsg(__FILE__, __LINE__))
    end if

    proc_ncell(:) = 0
    do cid = 1,nclumps
       proc_ncell(clumps(cid)%owner) = proc_ncell(clumps(cid)%owner) + clumps(cid)%ncells
    enddo

    ! determine offset (begg) for all processes,
    ! and then procinfo%begg and procinfo%endg (for iam)
    allocate(proc_begg(0:npes-1), stat=ier)
    if (ier /= 0) then
       write(iulog,*) 'decompInit_lnd(): allocation error for proc_begg'
       ! call_endrun(msg=errMsg(__FILE__, __LINE__))
    end if

    proc_begg(0) = 1
    do pid = 1,npes-1
       proc_begg(pid) = proc_begg(pid-1) + proc_ncell(pid-1)
    enddo
    procinfo%begg = proc_begg(iam)
    procinfo%endg = (procinfo%begg-1) + procinfo%ncells

    ! determine offset for each clump assigned to each process
    ! (re-using proc_begg as work space)
    do cid = 1,nclumps
      clumps(cid)%begg = proc_begg(clumps(cid)%owner)
      proc_begg(clumps(cid)%owner) = proc_begg(clumps(cid)%owner) &
                                   + clumps(cid)%ncells
      clumps(cid)%endg = proc_begg(clumps(cid)%owner) - 1
    enddo

    ! free work space
    deallocate(proc_ncell, proc_begg)

    ! Set ldecomp

    allocate(ldecomp%gdc2glo(numg), stat=ier)
    if (ier /= 0) then
       write(iulog,*) 'decompInit_lnd(): allocation error1 for ldecomp, etc'
       !! call_endrun(msg=errMsg(__FILE__, __LINE__))
    end if
    allocate(clumpcnt(nclumps),stat=ier)
    if (ier /= 0) then
       write(iulog,*) 'decompInit_lnd(): allocation error1 for clumpcnt'
       !! call_endrun(msg=errMsg(__FILE__, __LINE__))
    end if

    ldecomp%gdc2glo(:) = 0
    ag = 0

    ! clumpcnt is the start gdc index of each clump

    clumpcnt = 0
    do cid = 1,nclumps
       clumpcnt(cid) = clumps(cid)%begg
    enddo

    ! now go through gridcells one at a time and increment clumpcnt
    ! in order to set gdc2glo

    do aj = 1,lnj
    do ai = 1,lni
       an = (aj-1)*lni + ai
       cid = lcid(an)
       if (cid > 0) then
          ag = clumpcnt(cid)
          ldecomp%gdc2glo(ag) = an
          clumpcnt(cid) = clumpcnt(cid) + 1
       end if
    end do
    end do

    deallocate(clumpcnt)

    ! Set gsMap_lnd_gdc2glo (the global index here includes mask=0 or ocean points)

    call get_proc_bounds(beg, end)
    allocate(gindex(beg:end))
    do n = beg,end
       gindex(n) = ldecomp%gdc2glo(n)
    enddo
    lsize = end-beg+1
    gsize = lni * lnj
    !call mct_gsMap_init(gsMap_lnd_gdc2glo, gindex, mpicom, comp_id, lsize, gsize)
    deallocate(gindex)

    ! Diagnostic output

    if (masterproc) then
       write(iulog,*)' Surface Grid Characteristics'
       write(iulog,*)'   longitude points               = ',lni
       write(iulog,*)'   latitude points                = ',lnj
       write(iulog,*)'   total number of land gridcells = ',numg
       write(iulog,*)' Decomposition Characteristics'
       write(iulog,*)'   clumps per process             = ',clump_pproc
       write(iulog,*)' gsMap Characteristics'
       !write(iulog,*) '  lnd gsmap glo num of segs      = ',mct_gsMap_ngseg(gsMap_lnd_gdc2glo)
       write(iulog,*)
    end if


  end subroutine decompInit_lnd_simple

  !------------------------------------------------------------------------------
  subroutine decompInit_clumps(glcmask)
    !
    ! !DESCRIPTION:
    ! This subroutine initializes the land surface decomposition into a clump
    ! data structure.  This assumes each pe has the same number of clumps
    ! set by clump_pproc
    !
    ! !USES:
    use subgridMod, only : subgrid_get_gcellinfo
    !
    ! !ARGUMENTS:
    implicit none
    integer , pointer, optional   :: glcmask(:)  ! glc mask
    !integer , pointer, optional   :: num_tunits_per_grd(:)  ! Number of topounits per grid
    !
    ! !LOCAL VARIABLES:
    integer :: ln,an              ! indices
    integer :: i,g,l,k            ! indices
    integer :: cid,pid            ! indices
    integer :: n,m,np             ! indices
    integer :: anumg              ! lnd num gridcells
    integer :: icells             ! temporary
    integer :: begg, endg         ! temporary
    integer :: itunits            ! temporary
    integer :: ilunits            ! temporary
    integer :: icols              ! temporary
    integer :: ipfts              ! temporary
    integer :: icohorts           ! temporary
    integer :: ier                ! error code
    integer :: glev, tlev, llev, clev, plev, hlev  ! order of subgrid levels in the allvec arrays
    integer :: nlev               ! number of subgrid levels
    integer, allocatable :: allvecg(:,:)  ! temporary vector "global"
    integer, allocatable :: allvecl(:,:)  ! temporary vector "local"
    integer, allocatable :: proc_nXXX(:) ! number of XXX assigned to a process
    integer, allocatable :: proc_begX(:) ! beginning XXX index assigned to a process
    integer :: ntest
    character(len=32), parameter :: subname = 'decompInit_clumps'
    !------------------------------------------------------------------------------
    
    !--- assign order of subgrid levels in allvecl and allvecg arrays ---
    nlev=6  ! number of subgrid levels
    glev=1  ! gridcell
    tlev=2  ! topounit
    llev=3  ! landunit
    clev=4  ! column
    plev=5  ! pft/patch
    hlev=6  ! cohort

    !--- assign gridcells to clumps (and thus pes) ---
    call get_proc_bounds(begg, endg)

    allocate(allvecl(nclumps,nlev))   ! local  clumps [gcells,topounits,lunits,cols,pfts,cohs]
    allocate(allvecg(nclumps,nlev))   ! global clumps [gcells,topounits,lunits,cols,pfts,cohs]

    ! Determine the number of gridcells, topounits, landunits, columns, pfts, and cohorts 
    ! on this processor 
    ! Determine number of topounits, landunits, columns and pfts for each global
    ! gridcell index (an) that is associated with the local gridcell index (ln)
    ! More detail: an is the row-major order 1d-index into the global ixj grid.

    itunits=0
    ilunits=0
    icols=0
    ipfts=0
    icohorts=0 

    allvecg= 0
    allvecl= 0
    ! Loop through the gridcells on this proc
    do anumg = begg,endg
       ! an is the row-major order 1d-index into the global ixj grid.
       an  = ldecomp%gdc2glo(anumg)
       cid = lcid(an)
       ln  = anumg
       if(max_topounits > 1) then
          if (present(glcmask)) then
             call subgrid_get_gcellinfo (ln, ntunits=itunits, nlunits=ilunits, ncols=icols, npfts=ipfts, &
                 ncohorts=icohorts, glcmask=glcmask(ln), num_tunits_per_grd= ldomain%num_tunits_per_grd(ln))
          else
             call subgrid_get_gcellinfo (ln, ntunits=itunits, nlunits=ilunits, ncols=icols, npfts=ipfts, &
                 ncohorts=icohorts, num_tunits_per_grd= ldomain%num_tunits_per_grd(ln) )
          endif
       else 
          if (present(glcmask)) then
             call subgrid_get_gcellinfo (ln, ntunits=itunits, nlunits=ilunits, ncols=icols, npfts=ipfts, &
                 ncohorts=icohorts, glcmask=glcmask(ln))
          else
             call subgrid_get_gcellinfo (ln, ntunits=itunits, nlunits=ilunits, ncols=icols, npfts=ipfts, &
                 ncohorts=icohorts )
          endif
       endif
       
       allvecl(cid,glev) = allvecl(cid,glev) + 1           ! number of gridcells for local clump cid
       allvecl(cid,tlev) = allvecl(cid,tlev) + itunits     ! number of topographic units for local clump cid
       allvecl(cid,llev) = allvecl(cid,llev) + ilunits     ! number of landunits for local clump cid
       allvecl(cid,clev) = allvecl(cid,clev) + icols       ! number of columns for local clump cid
       allvecl(cid,plev) = allvecl(cid,plev) + ipfts       ! number of pfts for local clump cid 
       allvecl(cid,hlev) = allvecl(cid,hlev) + icohorts    ! number of cohorts for local clump cid 
    enddo
    !call mpi_allreduce(allvecl,allvecg,size(allvecg),MPI_INTEGER,MPI_SUM,mpicom,ier)

    ! Determine overall  total gridcells, landunits, columns and pfts and distribute
    ! gridcells over clumps

    numg = 0
    numt = 0
    numl = 0
    numc = 0
    nump = 0
    numCohort = 0

    do cid = 1,nclumps
       icells      = allvecg(cid,glev)  ! number of all clump cid gridcells (over all processors)
       itunits  = allvecg(cid,tlev)  ! number of all clump cid topounits (over all processors)
       ilunits     = allvecg(cid,llev)  ! number of all clump cid landunits (over all processors)
       icols       = allvecg(cid,clev)  ! number of all clump cid columns (over all processors)
       ipfts       = allvecg(cid,plev)  ! number of all clump cid pfts (over all processors)
       icohorts    = allvecg(cid,hlev)  ! number of all clump cid cohorts (over all processors)

       !--- overall total ---
       numg = numg + icells         ! total number of gridcells
       numt = numt + itunits     ! total number of landunits
       numl = numl + ilunits        ! total number of landunits
       numc = numc + icols          ! total number of columns
       nump = nump + ipfts          ! total number of pfts
       numCohort = numCohort + icohorts       ! total number of cohorts

       !--- give gridcell to cid ---
       clumps(cid)%ntunits  = clumps(cid)%ntunits  + itunits  
       clumps(cid)%nlunits     = clumps(cid)%nlunits  + ilunits  
       clumps(cid)%ncols       = clumps(cid)%ncols    + icols
       clumps(cid)%npfts       = clumps(cid)%npfts    + ipfts
       clumps(cid)%nCohorts    = clumps(cid)%nCohorts + icohorts

       !--- give gridcell to the proc that owns the cid ---
       if (iam == clumps(cid)%owner) then
          procinfo%ntunits  = procinfo%ntunits  + itunits
          procinfo%nlunits     = procinfo%nlunits  + ilunits
          procinfo%ncols       = procinfo%ncols    + icols
          procinfo%npfts       = procinfo%npfts    + ipfts
          procinfo%nCohorts    = procinfo%nCohorts + icohorts
       endif

    enddo

    allocate(proc_nXXX(0:npes-1), stat=ier)
    if (ier /= 0) then
       write(iulog,*) 'decompInit_clumps(): allocation error for proc_nXXX'
       ! call_endrun(msg=errMsg(__FILE__, __LINE__))
    end if

    allocate(proc_begX(0:npes-1), stat=ier)
    if (ier /= 0) then
       write(iulog,*) 'decompInit_clumps(): allocation error for proc_begX'
       ! call_endrun(msg=errMsg(__FILE__, __LINE__))
    end if

    ! TOPOUNITS:
    ! calculate number of topographic units per process
    proc_nXXX(:) = 0
    do cid = 1,nclumps
       proc_nXXX(clumps(cid)%owner) = &
        proc_nXXX(clumps(cid)%owner) + clumps(cid)%ntunits
    enddo

    ! determine offset (begt) for all processes,
    ! and then procinfo%begt and procinfo%endt (for iam)
    proc_begX(0) = 1
    do pid = 1,npes-1
       proc_begX(pid) = proc_begX(pid-1) + proc_nXXX(pid-1)
    enddo
    procinfo%begt = proc_begX(iam)
    procinfo%endt = (procinfo%begt-1) + procinfo%ntunits

    ! determine topounit offset for each clump assigned to each process
    ! (re-using proc_begX as work space)
    do cid = 1,nclumps
      clumps(cid)%begt = proc_begX(clumps(cid)%owner)
      proc_begX(clumps(cid)%owner) = proc_begX(clumps(cid)%owner) &
                                   + clumps(cid)%ntunits
      clumps(cid)%endt = proc_begX(clumps(cid)%owner) - 1
    enddo

    ! LUNITS:
    ! calculate number of lunits per process
    proc_nXXX(:) = 0
    do cid = 1,nclumps
       proc_nXXX(clumps(cid)%owner) = &
        proc_nXXX(clumps(cid)%owner) + clumps(cid)%nlunits
    enddo

    ! determine offset (begl) for all processes,
    ! and then procinfo%begl and procinfo%endl (for iam)
    proc_begX(0) = 1
    do pid = 1,npes-1
       proc_begX(pid) = proc_begX(pid-1) + proc_nXXX(pid-1)
    enddo
    procinfo%begl = proc_begX(iam)
    procinfo%endl = (procinfo%begl-1) + procinfo%nlunits

    ! determine lunit offset for each clump assigned to each process
    ! (re-using proc_begX as work space)
    do cid = 1,nclumps
      clumps(cid)%begl = proc_begX(clumps(cid)%owner)
      proc_begX(clumps(cid)%owner) = proc_begX(clumps(cid)%owner) &
                                   + clumps(cid)%nlunits
      clumps(cid)%endl = proc_begX(clumps(cid)%owner) - 1
    enddo

    ! COLS:
    ! calculate number of cols per process
    proc_nXXX(:) = 0
    do cid = 1,nclumps
       proc_nXXX(clumps(cid)%owner) = &
        proc_nXXX(clumps(cid)%owner) + clumps(cid)%ncols
    enddo

    ! determine offset (begc) for all processes,
    ! and then procinfo%begc and procinfo%endc (for iam)
    proc_begX(0) = 1
    do pid = 1,npes-1
       proc_begX(pid) = proc_begX(pid-1) + proc_nXXX(pid-1)
    enddo
    procinfo%begc = proc_begX(iam)
    procinfo%endc = (procinfo%begc-1) + procinfo%ncols

    ! determine col offset for each clump assigned to each process
    ! (re-using proc_begX as work space)
    do cid = 1,nclumps
      clumps(cid)%begc = proc_begX(clumps(cid)%owner)
      proc_begX(clumps(cid)%owner) = proc_begX(clumps(cid)%owner) &
                                   + clumps(cid)%ncols
      clumps(cid)%endc = proc_begX(clumps(cid)%owner) - 1
    enddo

    ! PFTS:
    ! calculate number of pfts per process
    proc_nXXX(:) = 0
    do cid = 1,nclumps
       proc_nXXX(clumps(cid)%owner) = &
        proc_nXXX(clumps(cid)%owner) + clumps(cid)%npfts
    enddo

    ! determine offset (begp) for all processes,
    ! and then procinfo%begp and procinfo%endp (for iam)
    proc_begX(0) = 1
    do pid = 1,npes-1
       proc_begX(pid) = proc_begX(pid-1) + proc_nXXX(pid-1)
    enddo
    procinfo%begp = proc_begX(iam)
    procinfo%endp = (procinfo%begp-1) + procinfo%npfts

    ! determine col offset for each clump assigned to each process
    ! (re-using proc_begX as work space)
    do cid = 1,nclumps
      clumps(cid)%begp = proc_begX(clumps(cid)%owner)
      proc_begX(clumps(cid)%owner) = proc_begX(clumps(cid)%owner) &
                                   + clumps(cid)%npfts
      clumps(cid)%endp = proc_begX(clumps(cid)%owner) - 1
    enddo

    ! COHORTS:
    ! calculate number of cohorts per process
    proc_nXXX(:) = 0
    do cid = 1,nclumps
       proc_nXXX(clumps(cid)%owner) = &
        proc_nXXX(clumps(cid)%owner) + clumps(cid)%nCohorts
    enddo

    ! determine offset (begCohort) for all processes,
    ! and then procinfo%begCohort and procinfo%endCohort (for iam)
    proc_begX(0) = 1
    do pid = 1,npes-1
       proc_begX(pid) = proc_begX(pid-1) + proc_nXXX(pid-1)
    enddo
    procinfo%begCohort = proc_begX(iam)
    procinfo%endCohort = (procinfo%begCohort-1) + procinfo%nCohorts

    ! determine col offset for each clump assigned to each process
    ! (re-using proc_begX as work space)
    do cid = 1,nclumps
      clumps(cid)%begCohort = proc_begX(clumps(cid)%owner)
      proc_begX(clumps(cid)%owner) = proc_begX(clumps(cid)%owner) &
                                   + clumps(cid)%nCohorts
      clumps(cid)%endCohort = proc_begX(clumps(cid)%owner) - 1
    enddo

    ! free work space
    deallocate(proc_nXXX, proc_begX)

    do n = 1,nclumps
       if (clumps(n)%ncells      /= allvecg(n,glev) .or. &
           clumps(n)%ntunits  /= allvecg(n,tlev) .or. &
           clumps(n)%nlunits     /= allvecg(n,llev) .or. &
           clumps(n)%ncols       /= allvecg(n,clev) .or. &
           clumps(n)%npfts       /= allvecg(n,plev) .or. &
           clumps(n)%nCohorts    /= allvecg(n,hlev)) then

               write(iulog,*) 'decompInit_glcp(): allvecg error ncells ',iam,n,clumps(n)%ncells ,allvecg(n,glev)
               write(iulog,*) 'decompInit_glcp(): allvecg error topounits ',iam,n,clumps(n)%ntunits,allvecg(n,tlev)
               write(iulog,*) 'decompInit_glcp(): allvecg error lunits ',iam,n,clumps(n)%nlunits,allvecg(n,llev)
               write(iulog,*) 'decompInit_glcp(): allvecg error ncols  ',iam,n,clumps(n)%ncols  ,allvecg(n,clev)
               write(iulog,*) 'decompInit_glcp(): allvecg error pfts   ',iam,n,clumps(n)%npfts  ,allvecg(n,plev)
               write(iulog,*) 'decompInit_glcp(): allvecg error cohorts ',iam,n,clumps(n)%nCohorts ,allvecg(n,hlev)

               ! call_endrun(msg=errMsg(__FILE__, __LINE__))

       endif
    enddo

    deallocate(allvecg,allvecl)
    deallocate(lcid)

  end subroutine decompInit_clumps


end module decompInitMod
