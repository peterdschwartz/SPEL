from django.contrib.auth import get_user_model
from django.db import models
from django.db.models import UniqueConstraint
from django.utils import timezone

from .utils.helper import compute_config_hash


class Modules(models.Model):
    objects = models.Manager()
    module_id = models.AutoField(primary_key=True)
    module_name = models.CharField(unique=True, max_length=100)

    class Meta:
        db_table = "modules"


class ModuleDependency(models.Model):
    objects = models.Manager()
    dependency_id = models.AutoField(primary_key=True)
    module = models.ForeignKey("Modules", on_delete=models.CASCADE)
    dep_module = models.ForeignKey(
        "Modules",
        on_delete=models.CASCADE,
        related_name="moduledependency_dep_module_set",
    )
    object_used = models.CharField(max_length=100)

    class Meta:
        db_table = "module_dependency"
        constraints = [
            UniqueConstraint(
                fields=("module", "dep_module", "object_used"), name="unique_mod_dep"
            ),
        ]


class UserTypes(models.Model):
    objects = models.Manager()
    user_type_id = models.AutoField(primary_key=True)
    module = models.ForeignKey(
        "Modules",
        on_delete=models.CASCADE,
        related_name="user_type_module",
    )
    user_type_name = models.CharField(unique=True, max_length=100)

    class Meta:
        db_table = "user_types"
        constraints = [
            UniqueConstraint(fields=("module", "user_type_name"), name="unique_types")
        ]


class TypeDefinitions(models.Model):
    objects = models.Manager()
    define_id = models.AutoField(primary_key=True)
    type_module = models.ForeignKey(
        "Modules",
        on_delete=models.CASCADE,
        related_name="type_def_module",
    )
    user_type = models.ForeignKey(
        "UserTypes",
        on_delete=models.CASCADE,
        related_name="user_type_def",
    )
    member_type = models.CharField(max_length=100)
    member_name = models.CharField(max_length=100)
    dim = models.IntegerField()
    bounds = models.CharField(max_length=100)

    class Meta:
        db_table = "type_definitions"
        unique_together = (("user_type", "member_type", "member_name", "type_module"),)


class UserTypeInstances(models.Model):
    objects = models.Manager()
    instance_id = models.AutoField(primary_key=True)
    inst_module = models.ForeignKey(
        "Modules",
        on_delete=models.CASCADE,
        related_name="instance_module",
    )
    instance_type = models.ForeignKey(
        "UserTypes",
        on_delete=models.CASCADE,
        related_name="instance_type",
    )
    instance_name = models.CharField(max_length=100)

    class Meta:
        db_table = "user_type_instances"
        constraints = [
            UniqueConstraint(
                fields=("inst_module", "instance_type", "instance_name"),
                name="unique_instances",
            )
        ]


class Subroutines(models.Model):
    objects = models.Manager()
    subroutine_id = models.AutoField(primary_key=True)
    subroutine_name = models.CharField(max_length=100)
    module = models.ForeignKey(
        "Modules",
        on_delete=models.CASCADE,
        related_name="subroutine_module",
    )

    class Meta:
        db_table = "subroutines"
        constraints = [
            UniqueConstraint(fields=("subroutine_name", "module"), name="unique_subs")
        ]

    def __str__(self):
        return f"{self.subroutine_name}"


class SubroutineArgs(models.Model):
    objects = models.Manager()
    arg_id = models.AutoField(primary_key=True)
    subroutine = models.ForeignKey(
        "Subroutines",
        models.CASCADE,
        related_name="subroutine_args",
    )
    arg_type = models.CharField(max_length=100)
    arg_name = models.CharField(max_length=100)
    dim = models.IntegerField()

    class Meta:
        db_table = "subroutine_args"
        constraints = [
            UniqueConstraint(
                fields=("subroutine", "arg_type", "arg_name", "dim"),
                name="unique_sub_args",
            ),
        ]


class SubroutineCalltree(models.Model):
    objects = models.Manager()
    parent_id = models.AutoField(primary_key=True)
    parent_subroutine = models.ForeignKey(
        "Subroutines",
        on_delete=models.CASCADE,
        related_name="parent_subroutine",
    )
    child_subroutine = models.ForeignKey(
        "Subroutines",
        on_delete=models.CASCADE,
        related_name="child_subroutine",
    )

    lineno = models.IntegerField()

    class Meta:
        db_table = "subroutine_calltree"
        constraints = [
            UniqueConstraint(
                fields=("parent_subroutine", "child_subroutine", "lineno"),
                name="unique_calltree",
            ),
        ]


class SubroutineLocalArrays(models.Model):
    objects = models.Manager()
    local_arry_id = models.AutoField(primary_key=True)
    subroutine = models.ForeignKey(
        "Subroutines",
        on_delete=models.CASCADE,
        related_name="subroutine_locals",
    )
    array_name = models.CharField(max_length=100)
    dim = models.IntegerField()

    class Meta:
        db_table = "subroutine_local_arrays"
        constraints = [
            UniqueConstraint(
                fields=("subroutine", "array_name"), name="unique_sub_locals"
            ),
        ]


class SubroutineActiveGlobalVars(models.Model):
    objects = models.Manager()
    variable_id = models.AutoField(primary_key=True)
    subroutine = models.ForeignKey(
        "Subroutines",
        on_delete=models.CASCADE,
        related_name="subroutine_dtype_vars",
    )
    instance = models.ForeignKey(
        "UserTypeInstances",
        on_delete=models.CASCADE,
        related_name="active_instances",
    )
    member = models.ForeignKey(
        "TypeDefinitions",
        on_delete=models.CASCADE,
        related_name="active_member",
    )
    status = models.CharField(max_length=2)
    ln = models.IntegerField(default=-1)

    class Meta:
        db_table = "subroutine_active_global_vars"
        constraints = [
            UniqueConstraint(
                fields=("subroutine", "instance", "member", "status", "ln"),
                name="unique_sub_dtype",
            )
        ]


class IntrinsicGlobals(models.Model):
    objects = models.Manager()
    var_id = models.AutoField(primary_key=True)
    gv_module = models.ForeignKey(
        "Modules",
        on_delete=models.CASCADE,
        related_name="global_var",
    )
    dim = models.IntegerField()
    var_type = models.CharField(max_length=50)
    var_name = models.CharField(max_length=100)
    bounds = models.CharField(max_length=100)
    value = models.TextField()

    class Meta:
        db_table = "intrinsic_globals"
        constraints = [
            UniqueConstraint(
                fields=(
                    "gv_module",
                    "var_name",
                ),
                name="unique_globals",
            )
        ]
        indexes = [
            models.Index(fields=("gv_module", "var_name")),
        ]


class SubroutineIntrinsicGlobals(models.Model):
    objects = models.Manager()
    sub_gv_id = models.AutoField(primary_key=True)
    gv_id = models.ForeignKey(
        "IntrinsicGlobals",
        on_delete=models.CASCADE,
        related_name="active_intrinsic",
    )
    sub_id = models.ForeignKey(
        "Subroutines",
        on_delete=models.CASCADE,
        related_name="subroutine_global_vars",
    )
    gv_ln = models.IntegerField(default=-1)

    class Meta:
        db_table = "subroutine_intrinsic_globals"
        constraints = [
            UniqueConstraint(fields=("gv_id", "sub_id"), name="unique_sub_intrinsic")
        ]


class NamelistVariable(models.Model):
    objects = models.Manager()
    nml_id = models.AutoField(primary_key=True)
    active_var_id = models.ForeignKey(
        "IntrinsicGlobals",
        on_delete=models.CASCADE,
        related_name="namelist",
    )

    class Meta:
        db_table = "namelist_variables"
        constraints = [
            UniqueConstraint(fields=("active_var_id",), name="unique_nml_var")
        ]


class FlatIf(models.Model):
    objects = models.Manager()
    flatif_id = models.AutoField(primary_key=True)

    subroutine = models.ForeignKey(
        "Subroutines",
        on_delete=models.CASCADE,
        related_name="flat_ifs",
    )

    start_ln = models.IntegerField()
    end_ln = models.IntegerField()
    condition = models.TextField()
    active = models.BooleanField()

    class Meta:
        db_table = "flat_if_blocks"
        constraints = [
            UniqueConstraint(
                fields=(
                    "flatif_id",
                    "start_ln",
                    "subroutine",
                    "end_ln",
                ),
                name="unique_ifs",
            )
        ]


class FlatIfNamelistVar(models.Model):
    objects = models.Manager()
    id = models.AutoField(primary_key=True)
    flatif = models.ForeignKey(FlatIf, on_delete=models.CASCADE)
    namelist_var = models.ForeignKey(NamelistVariable, on_delete=models.CASCADE)

    class Meta:
        db_table = "flat_if_namelist"
        constraints = [
            UniqueConstraint(
                fields=("id", "flatif", "namelist_var"), name="unique_flatif_var"
            )
        ]


User = get_user_model()

class ConfigProfile(models.Model):
    objects = models.Manager()
    owner = models.ForeignKey(User, on_delete=models.CASCADE, related_name="configs")
    name = models.CharField(max_length=128)
    data = models.JSONField(default=dict)
    user_hash = models.CharField(max_length=64, db_index=True, blank=True)
    created_at = models.DateTimeField(default=timezone.now)

    class Meta:
        db_table = "config_profile"
        indexes = [models.Index(fields=["owner", "user_hash"])]
        constraints = [
            models.UniqueConstraint(
                fields=["owner", "name"], name="uniq_configprofile_name_per_owner"
            ),
        ]

    def save(self, *args, **kwargs):
        self.config_hash = compute_config_hash(self.data)
        super().save(*args, **kwargs)


class PresetConfig(models.Model):
    objects = models.Manager()
    slug = models.SlugField(unique=True)
    name = models.CharField(max_length=128)
    data = models.JSONField(default=dict)
    preset_hash = models.CharField(max_length=64, db_index=True, blank=True)
    is_default = models.BooleanField(default=False)

    class Meta:
        db_table = "preset_configs"
        indexes = [models.Index(fields=["preset_hash"])]

    def save(self, *args, **kwargs):
        self.config_hash = compute_config_hash(self.data)
        super().save(*args, **kwargs)


class IfEvaluationByHash(models.Model):
    objects = models.Manager()
    config_hash = models.CharField(max_length=64, db_index=True)
    flatif = models.ForeignKey("FlatIf", on_delete=models.CASCADE)
    subroutine = models.ForeignKey("Subroutines", on_delete=models.CASCADE, db_index=True)
    start_ln = models.IntegerField()
    end_ln   = models.IntegerField()
    is_active = models.BooleanField()

    class Meta:
        db_table = "if_eval_by_hash"
        constraints = [
            models.UniqueConstraint(fields=["config_hash", "flatif"], name="uniq_if_eval_hash_flatif")
        ]
        indexes = [
            models.Index(fields=["config_hash", "subroutine", "is_active", "start_ln", "end_ln"]),
        ]

