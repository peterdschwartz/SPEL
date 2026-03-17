
    module emulator_setup_mod
      use iso_fortran_env, only : real64, int32
      use emulator_mod
      implicit none
      integer, parameter :: rkind = real64, ikind = int32
    contains


subroutine create_emulator_fields(      canopystate_vars,&
  col_cf,&
  soillittverttranspparamsinst,&
  cnstate_vars,&
  col_nf,&
  col_pf,&
  col_ns,&
  decomp_cascade_con,&
  col_ps,&
  col_cs,&
  emulator, filter)
  use cndecompcascadecontype, only: decomp_cascade_type
  use columndatatype, only: column_phosphorus_flux
  use canopystatetype, only: canopystate_type
  use soillittverttranspmod, only: soillittverttranspparamstype
  use cnstatetype, only: cnstate_type
  use columndatatype, only: column_phosphorus_state
  use columndatatype, only: column_carbon_flux
  use columndatatype, only: column_nitrogen_flux
  use columndatatype, only: column_carbon_state
  use columndatatype, only: column_nitrogen_state
  use filtermod, only : clumpfilter

  type(soillittverttranspparamstype), intent(in) :: soillittverttranspparamsinst
  type(decomp_cascade_type), intent(in) :: decomp_cascade_con
  type(canopystate_type), intent(in) :: canopystate_vars
  type(cnstate_type), intent(in) :: cnstate_vars
  type(column_carbon_state), intent(in) :: col_cs
  type(column_nitrogen_state), intent(in) :: col_ns
  type(column_phosphorus_state), intent(in) :: col_ps
  type(column_carbon_flux), intent(in) :: col_cf
  type(column_nitrogen_flux), intent(in) :: col_nf
  type(column_phosphorus_flux), intent(in) :: col_pf
   type(emulator_t), intent(inout) :: emulator
   type(clumpfilter), intent(in)    :: filter


    type(field_list_t) :: in_list, out_list

    ! Build input list in the exact order used during training
        call in_list%add(canopystate_vars%altmax_col, filter%num_soilc, filter%soilc)
    call in_list%add(col_cf%decomp_cpools_sourcesink, filter%num_soilc, filter%soilc)
    call in_list%add(soillittverttranspparamsinst%cryoturb_diffusion_k, filter%num_soilc, filter%soilc)
    call in_list%add(canopystate_vars%altmax_lastyear_col, filter%num_soilc, filter%soilc)
    call in_list%add(cnstate_vars%scalaravg_col, filter%num_soilc, filter%soilc)
    call in_list%add(col_nf%decomp_npools_sourcesink, filter%num_soilc, filter%soilc)
    call in_list%add(col_pf%decomp_ppools_sourcesink, filter%num_soilc, filter%soilc)
    call in_list%add(col_ns%decomp_npools_vr, filter%num_soilc, filter%soilc)
    call in_list%add(decomp_cascade_con%is_cwd, filter%num_soilc, filter%soilc)
    call in_list%add(decomp_cascade_con%spinup_factor, filter%num_soilc, filter%soilc)
    call in_list%add(soillittverttranspparamsinst%som_diffus, filter%num_soilc, filter%soilc)
    call in_list%add(soillittverttranspparamsinst%max_altdepth_cryoturbation, filter%num_soilc, filter%soilc)
    call in_list%add(col_ps%decomp_ppools_vr, filter%num_soilc, filter%soilc)
    call in_list%add(col_cs%decomp_cpools_vr, filter%num_soilc, filter%soilc)

    ! Build output list in the exact order used during training
        call out_list%add(col_cf%decomp_cpools_transport_tendency,filter%num_soilc, filter%soilc)
    call out_list%add(cnstate_vars%som_adv_coef_col,filter%num_soilc, filter%soilc)
    call out_list%add(cnstate_vars%som_diffus_coef_col,filter%num_soilc, filter%soilc)
    call out_list%add(col_pf%decomp_ppools_transport_tendency,filter%num_soilc, filter%soilc)
    call out_list%add(col_ns%decomp_npools_vr,filter%num_soilc, filter%soilc)
    call out_list%add(col_nf%decomp_npools_transport_tendency,filter%num_soilc, filter%soilc)
    call out_list%add(col_ps%decomp_ppools_vr,filter%num_soilc, filter%soilc)
    call out_list%add(col_cs%decomp_cpools_vr,filter%num_soilc, filter%soilc)

      if (.not. emulator%initialized) then
         call emulator%init_from_field_lists("spel_emulator_traced.pt", in_list, out_list)
      end if

    end subroutine create_emulator_fields


    end module emulator_setup_mod
