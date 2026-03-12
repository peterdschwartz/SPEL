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

    def __str__(self):
        return f"Module(name={self.module_name})"


class ModuleDependency(models.Model):
    objects = models.Manager()
    dependency_id = models.AutoField(primary_key=True)
    module = models.ForeignKey(Modules, on_delete=models.CASCADE)
    dep_module = models.ForeignKey(
        Modules,
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
        Modules,
        on_delete=models.CASCADE,
        related_name="user_type_module",
    )
    user_type_name = models.CharField(unique=True, max_length=100)

    class Meta:
        db_table = "user_types"
        constraints = [
            UniqueConstraint(fields=("module", "user_type_name"), name="unique_types")
        ]

    def __str__(self):
        return f"UserType(mod={self.module.module_name},type={self.user_type_name})"


class TypeDefinitions(models.Model):
    objects = models.Manager()
    define_id = models.AutoField(primary_key=True)
    type_module = models.ForeignKey(
        Modules,
        on_delete=models.CASCADE,
        related_name="type_def_module",
    )
    user_type = models.ForeignKey(
        UserTypes,
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
        Modules,
        on_delete=models.CASCADE,
        related_name="instance_module",
    )
    instance_type = models.ForeignKey(
        UserTypes,
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

    def __str__(self):
        return f"UserTypeInstance(mod={self.inst_module.module_name},type={self.instance_type.user_type_name},inst={self.instance_name})"


class Subroutines(models.Model):
    objects = models.Manager()
    subroutine_id = models.AutoField(primary_key=True)
    subroutine_name = models.CharField(max_length=100)
    module = models.ForeignKey(
        Modules,
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
        Subroutines,
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
                fields=("subroutine", "arg_name"),
                name="unique_sub_args",
            ),
        ]


class SubroutineCalltree(models.Model):
    objects = models.Manager()
    parent_id = models.AutoField(primary_key=True)
    parent_subroutine = models.ForeignKey(
        Subroutines,
        on_delete=models.CASCADE,
        related_name="parent_subroutine",
    )
    child_subroutine = models.ForeignKey(
        Subroutines,
        on_delete=models.CASCADE,
        related_name="child_subroutine",
    )
    args = models.TextField(default="")
    lineno = models.IntegerField()

    class Meta:
        db_table = "subroutine_calltree"
        constraints = [
            UniqueConstraint(
                fields=("parent_subroutine", "child_subroutine", "lineno"),
                name="unique_calltree",
            ),
        ]
        indexes = [
            models.Index(fields=["parent_subroutine"]),
            models.Index(fields=["parent_subroutine", "lineno"]),
        ]


class SubroutineLocalArrays(models.Model):
    objects = models.Manager()
    local_arry_id = models.AutoField(primary_key=True)
    subroutine = models.ForeignKey(
        Subroutines,
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
        Subroutines,
        on_delete=models.CASCADE,
        related_name="subroutine_dtype_vars",
    )
    instance = models.ForeignKey(
        UserTypeInstances,
        on_delete=models.CASCADE,
        related_name="active_instances",
    )
    member = models.ForeignKey(
        TypeDefinitions,
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


class SubroutineElmtypesByConfig(models.Model):
    objects = models.Manager()
    access_id = models.AutoField(primary_key=True)
    config_hash = models.CharField(max_length=64, db_index=True)
    subroutine = models.ForeignKey(
        Subroutines,
        on_delete=models.CASCADE,
        related_name="subroutine_dtype_vars_by_config",
    )
    instance = models.ForeignKey(
        UserTypeInstances,
        on_delete=models.CASCADE,
        related_name="active_instances_by_config",
        null=True,
        blank=True,
    )
    member = models.ForeignKey(
        TypeDefinitions,
        on_delete=models.CASCADE,
        related_name="active_members_by_config",
        null=True,
        blank=True,
    )

    status = models.CharField(max_length=2)
    ln = models.IntegerField(default=-1)
    var_name = models.CharField(max_length=255, default="", db_index=True)
    member_path = models.CharField(max_length=255, default="", db_index=True)
    created_at = models.DateTimeField(auto_now_add=True)
    last_used_at = models.DateTimeField(default=timezone.now, db_index=True)
    indirect = models.BooleanField(default=False)

    class Meta:
        db_table = "elmtype_access_for_config"
        indexes = [
            models.Index(fields=["config_hash", "subroutine"]),
            models.Index(fields=["last_used_at"]),
        ]
        constraints = [
            UniqueConstraint(
                fields=(
                    "subroutine",
                    "var_name",
                    "member_path",
                    "status",
                    "ln",
                    "config_hash",
                ),
                name="unique_sub_dtype_by_config",
            )
        ]

    def __str__(self)->str:
        var_name = f"{self.var_name}%{self.member_path}" if self.member_path else f"{self.var_name}"
        str_ = f"{var_name} {self.status}@L{self.ln} in {self.subroutine.subroutine_name}"
        return str_
    
    def __eq__(self,other)-> bool:
        if not isinstance(other,SubroutineElmtypesByConfig): 
            return False
        return (
            self.subroutine.subroutine_id == other.subroutine.subroutine_id and
            self.instance.instance_id == other.instance.instance_id and
            self.member_path == other.member_path and
            self.var_name == other.var_name and
            self.status == other.status and 
            self.ln == other.ln and 
            self.config_hash == other.config_hash
        )

class ArgAccess(models.Model):
    objects = models.Manager()
    arg_access_id = models.AutoField(primary_key=True)
    subroutine = models.ForeignKey(Subroutines, on_delete=models.CASCADE)
    arg = models.ForeignKey(SubroutineArgs, on_delete=models.CASCADE)
    ln = models.IntegerField()
    # 'R', 'W', 'RW' (modify/read+write) — same codes you already use
    status = models.CharField(max_length=2)
    # field1%subfield%...
    member_path = models.CharField(max_length=255)

    class Meta:
        db_table = "arg_access"
        indexes = [
            models.Index(fields=["subroutine", "arg", "ln"]),
            models.Index(fields=["subroutine", "arg", "status"]),
        ]
        constraints = [
            models.UniqueConstraint(
                fields=("subroutine", "arg", "ln", "status", "member_path"),
                name="unique_arg_use_row",
            )
        ]


class CallsiteBinding(models.Model):
    objects = models.Manager()
    binding_id = models.AutoField(primary_key=True)
    parent_subroutine = models.ForeignKey(
        Subroutines,
        on_delete=models.CASCADE,
        related_name="as_binding_parent",
    )
    call = models.ForeignKey(
        SubroutineCalltree,
        on_delete=models.CASCADE,
        related_name="bindings",
    )
    dummy_arg = models.ForeignKey(SubroutineArgs, on_delete=models.CASCADE)

    from_elmtype = models.ForeignKey(
        UserTypeInstances,
        on_delete=models.CASCADE,
        related_name="bound_inst",
        null=True,
        blank=True,
    )
    from_arg = models.ForeignKey(
        SubroutineArgs,
        on_delete=models.CASCADE,
        related_name="bound_arg",
        null=True,
        blank=True,
    )
    from_local = models.ForeignKey(
        SubroutineLocalArrays,
        on_delete=models.CASCADE,
        related_name="bound_local",
        null=True,
        blank=True,
    )
    first_member = models.ForeignKey(
        TypeDefinitions,
        on_delete=models.CASCADE,
        related_name="bound_member",
        null=True,
        blank=True,
    )
    nested_level = models.IntegerField()
    scope = models.CharField(max_length=32)  # ELMTYPE, ARG, LOCAL
    var_name = models.CharField(max_length=255)
    member_path_str = models.CharField(max_length=255, default="", db_index=True)

    class Meta:
        db_table = "callsite_binding"
        constraints = [
            models.UniqueConstraint(
                fields=(
                    "call",
                    "dummy_arg",
                    "var_name",
                    "member_path_str",
                    "nested_level",
                ),
                name="uniq_binding_per_arg",
            )
        ]
        indexes = [
            models.Index(fields=["call"]),
            models.Index(fields=["dummy_arg"]),
            models.Index(fields=["parent_subroutine", "var_name"]),
        ]

    def __str__(self):
        return (
            f"{self.parent_subroutine.subroutine_name}->"
            + f"{self.call.child_subroutine.subroutine_name}@L{self.call.lineno} Scope={self.scope} via {self.dummy_arg.arg_name}"
        )


class PropagatedEffectByLn(models.Model):
    objects = models.Manager()
    prop_id = models.AutoField(primary_key=True)
    call_site = models.ForeignKey(
        SubroutineCalltree,
        on_delete=models.CASCADE,
        related_name="propagated_call_site",
    )
    binding = models.ForeignKey(
        CallsiteBinding,
        on_delete=models.CASCADE,
        related_name="propagated_effects",
    )
    var_name = models.CharField(max_length=255)
    member_path = models.CharField(max_length=255)
    # whether the parent passes an ELMTYPE, ARG, or LOCAL as an argument
    scope = models.CharField(max_length=32)
    from_elmtype = models.ForeignKey(
        UserTypeInstances,
        on_delete=models.CASCADE,
        related_name="propagated_inst",
        null=True,
        blank=True,
    )
    from_arg = models.ForeignKey(
        SubroutineArgs,
        on_delete=models.CASCADE,
        related_name="propagated_arg",
        null=True,
        blank=True,
    )
    from_local = models.ForeignKey(
        SubroutineLocalArrays,
        on_delete=models.CASCADE,
        related_name="propagated_local",
        null=True,
        blank=True,
    )
    first_member = models.ForeignKey(
        TypeDefinitions,
        on_delete=models.CASCADE,
        related_name="propagated_member",
        null=True,
        blank=True,
    )
    status = models.CharField(max_length=2)  # status at the lineno
    lineno = models.IntegerField()  # Line No of inside the child subroutine

    class Meta:
        db_table = "propagated_access_by_ln"
        constraints = [
            models.UniqueConstraint(
                fields=(
                    "call_site",
                    "binding",
                    "var_name",
                    "member_path",
                    "lineno",
                    "status",
                    "scope",
                ),
                name="unique_prop",
            )
        ]

    def __str__(self):
        varname = (
            f"{self.var_name}%{self.member_path}"
            if self.member_path
            else f"{self.var_name}"
        )
        dummy_arg = self.binding.dummy_arg.arg_name
        parent_sub = self.binding.parent_subroutine.subroutine_name
        child_sub = self.binding.call.child_subroutine.subroutine_name
        return (
            f"{parent_sub}::CALL {child_sub}({dummy_arg}={varname}) Scope={self.scope}"
        )


class ArgSummaryByHash(models.Model):
    objects = models.Manager()
    config_hash = models.CharField(max_length=64, db_index=True)
    arg_access_id = models.AutoField(primary_key=True)
    subroutine = models.ForeignKey(Subroutines, on_delete=models.CASCADE)
    arg = models.ForeignKey(SubroutineArgs, on_delete=models.CASCADE)
    ln = models.IntegerField()
    status = models.CharField(max_length=2)
    # field1%subfield%...
    member_path = models.CharField(max_length=255)

    class Meta:
        db_table = "arg_access_by_hash"
        indexes = [
            models.Index(fields=["subroutine", "arg", "ln"]),
            models.Index(fields=["subroutine", "arg", "status"]),
        ]
        constraints = [
            models.UniqueConstraint(
                fields=("subroutine", "arg", "ln", "status", "member_path"),
                name="unique_arg_use_by_hash_row",
            )
        ]


class CallsitePropagatedEffectByHash(models.Model):
    objects = models.Manager()
    effect_id = models.AutoField(primary_key=True)
    config_hash = models.CharField(max_length=64, db_index=True)

    call = models.ForeignKey(SubroutineCalltree, on_delete=models.CASCADE)
    var_name = models.CharField(max_length=255, db_index=True)
    member_path_str = models.CharField(max_length=255, default="", db_index=True)
    kind = models.CharField(max_length=32)
    child_subroutine = models.ForeignKey(Subroutines, on_delete=models.CASCADE)
    child_ln = models.IntegerField()
    status = models.CharField(max_length=2)

    class Meta:
        db_table = "callsite_propagated_effect"
        indexes = [
            models.Index(fields=["var_name", "status"]),
            models.Index(fields=["var_name", "member_path_str"]),
            models.Index(fields=["call", "child_subroutine", "child_ln"]),
        ]


class IntrinsicGlobals(models.Model):
    objects = models.Manager()
    var_id = models.AutoField(primary_key=True)
    gv_module = models.ForeignKey(
        Modules,
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
        IntrinsicGlobals,
        on_delete=models.CASCADE,
        related_name="active_intrinsic",
    )
    sub_id = models.ForeignKey(
        Subroutines,
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
        IntrinsicGlobals,
        on_delete=models.CASCADE,
        related_name="namelist",
    )

    class Meta:
        db_table = "namelist_variables"
        constraints = [
            UniqueConstraint(fields=("active_var_id",), name="unique_nml_var")
        ]


class CascadeDependence(models.Model):
    """
    E.g. num_pcropp depends on use_crop.
    """

    objects = models.Manager()
    cascade_id = models.AutoField(primary_key=True)

    cascade_var = models.CharField(max_length=100)
    trigger_var = models.ForeignKey(
        NamelistVariable,
        on_delete=models.CASCADE,
        related_name="triggers",
    )

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=("cascade_var", "trigger_var"),
                name="unique_dependence",
            )
        ]


class CascadePair(models.Model):
    """
    A single mapping (nml_val → cascade_val)
    E.g. (.false. → 0), (.true. → 10)
    """

    objects = models.Manager()
    pair_id = models.AutoField(primary_key=True)

    dependence = models.ForeignKey(
        CascadeDependence,
        on_delete=models.CASCADE,
        related_name="pairs",
    )
    nml_val = models.CharField(max_length=64)
    cascade_val = models.CharField(max_length=64)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=("dependence", "nml_val", "cascade_val"), name="unique_pair"
            )
        ]


class FlatIf(models.Model):
    objects = models.Manager()
    flatif_id = models.AutoField(primary_key=True)

    subroutine = models.ForeignKey(
        Subroutines,
        on_delete=models.CASCADE,
        related_name="flat_ifs",
    )

    start_ln = models.IntegerField()
    end_ln = models.IntegerField()
    condition = models.JSONField()
    active = models.BooleanField()

    class Meta:
        db_table = "flat_if_blocks"
        constraints = [
            UniqueConstraint(
                fields=(
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
                fields=("flatif", "namelist_var"), name="unique_flatif_var"
            )
        ]


class FlatIFCascadeVar(models.Model):
    objects = models.Manager()
    cv_if_id = models.AutoField(primary_key=True)
    flatif_cv = models.ForeignKey(FlatIf, on_delete=models.CASCADE)
    cascade = models.ForeignKey(CascadeDependence, on_delete=models.CASCADE)

    class Meta:
        db_table = "flat_if_cascade"
        constraints = [
            UniqueConstraint(
                fields=("flatif_cv", "cascade"), name="uniq_flatif_cascade"
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
        self.user_hash = compute_config_hash(self.data)
        update_fields = kwargs.get("update_fields")
        if update_fields is not None:
            update_fields = set(update_fields)
            update_fields.add("preset_hash")
            kwargs["update_fields"] = update_fields
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
        self.preset_hash = compute_config_hash(self.data)
        update_fields = kwargs.get("update_fields")
        if update_fields is not None:
            update_fields = set(update_fields)
            update_fields.add("preset_hash")
            kwargs["update_fields"] = update_fields
        super().save(*args, **kwargs)


class IfEvaluationByHash(models.Model):
    objects = models.Manager()
    config_hash = models.CharField(max_length=64, db_index=True)
    flatif = models.ForeignKey(FlatIf, on_delete=models.CASCADE)
    subroutine = models.ForeignKey(
        Subroutines,
        on_delete=models.CASCADE,
        db_index=True,
    )
    start_ln = models.IntegerField()
    end_ln = models.IntegerField()
    is_active = models.BooleanField()

    class Meta:
        db_table = "if_eval_by_hash"
        constraints = [
            models.UniqueConstraint(
                fields=["config_hash", "flatif"], name="uniq_if_eval_hash_flatif"
            )
        ]
        indexes = [
            models.Index(
                fields=["config_hash", "subroutine", "is_active", "start_ln", "end_ln"]
            ),
        ]
