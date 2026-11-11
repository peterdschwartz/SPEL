module nc_allocMod
   ! Bounds of a captured variable, for spel_varMod::spel_var_allocate:
   !    call nc_desc(ncid, 'col_es__t_soisno', 2, spel_d)
   !    call spel_var_allocate(col_es%t_soisno, spel_d)
   ! The subgrid dimension is enlarged to hold spel_nsets copies of the file data.

   use netcdf
   use nc_io, only: check, nc_file_subgrid, spel_nsets
   use spel_varMod, only: spel_var_desc
   implicit none

   public

contains

   subroutine scale_subgrid_bounds(ncid, varid, ndim, lbs, ubs)
      ! Enlarge the subgrid dimension to hold spel_nsets copies of the file data
      integer, intent(in) :: ncid, varid, ndim
      integer, intent(in) :: lbs(ndim)
      integer, intent(inout) :: ubs(ndim)
      integer :: sd, nfile
      call nc_file_subgrid(ncid, varid, ndim, sd, nfile)
      if (sd > 0) ubs(sd) = lbs(sd) - 1 + spel_nsets*(ubs(sd) - lbs(sd) + 1)
   end subroutine scale_subgrid_bounds

   subroutine nc_desc(ncid, varname, ndim, d)
      ! d: the bounds of file variable varname (its lbounds/ubounds attributes), set 1
      ! scaled to spel_nsets sets along the subgrid dimension
      integer, intent(in) :: ncid
      character(len=*), intent(in) :: varname
      integer, intent(in) :: ndim
      type(spel_var_desc), intent(out) :: d
      integer :: varid, status

      d%name = varname
      d%rank = ndim
      if (ndim == 0) then
         d%has_bounds = .true.
         return
      end if
      call check(nf90_inq_varid(ncid, trim(varname), varid))
      status = nf90_get_att(ncid, varid, "lbounds", d%lb(1:ndim))
      if (status == nf90_noerr) status = nf90_get_att(ncid, varid, "ubounds", d%ub(1:ndim))
      if (status /= nf90_noerr) then
         print *, "Error - couldn't find l/ubounds of ", trim(varname)
         stop
      end if
      call scale_subgrid_bounds(ncid, varid, ndim, d%lb(1:ndim), d%ub(1:ndim))
      d%has_bounds = .true.
   end subroutine nc_desc

end module nc_allocMod
