from django.contrib.auth import views as auth_views
from django.urls import path

from . import views

urlpatterns = [
    path("", views.home, name="home"),
    path("modules_calltree", views.modules_calltree, name="modules_calltree"),
    path("sub_view/<str:sub_name>", views.sub_view, name="sub_view"),
    path("subcall/", views.subcall, name="subcall"),
    path("calltree/<str:sub_name>", views.render_calltree, name="calltree"),
    path("table/<str:table_name>", views.view_table, name="view_table"),
    path(
        "subroutine-details/<str:sub_name>/",
        views.subroutine_details,
        name="subroutine_details",
    ),
    path(
        "type-details/<str:type_name>/",
        views.type_details,
        name="type_details",
    ),
    path("trace/<str:key>/", views.trace_dtype_var, name="trace"),
    path("configs/", views.config_start, name="config_start"),
    path("configs/<int:config_id>/", views.config_editor, name="config_editor"),
    path(
        "configs/<int:config_id>/set/<int:var_id>/",
        views.set_config_value,
        name="set_config_value",
    ),
    path(
        "configs/<int:config_id>/recompute/",
        views.recompute_if_evals,
        name="recompute_if_evals",
    ),
    path(
        "configs/recompute/", views.recompute_if_evals, name="recompute_if_evals_global"
    ),
    path("accounts/login/", auth_views.LoginView.as_view(), name="login"),
    path("accounts/logout/", auth_views.LogoutView.as_view(), name="logout"),
    path("accounts/signup/", views.SignupView.as_view(), name="signup"),
    # urls.py
    path(
        "configs/select/preset/<slug:slug>/", views.select_preset, name="select_preset"
    ),
    path(
        "configs/select/user/<int:config_id>/",
        views.select_user_config,
        name="select_user_config",
    ),
    path("configs/picker/", views.config_picker, name="config_picker"),
    path("configs/select/", views.select_config_htmx, name="select_config_htmx"),
    path("configs/values/", views.nml_values, name="nml_values"),
]
