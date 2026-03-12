from django import forms

from .models import SubroutineActiveGlobalVars


class SubroutineActiveSearch(forms.ModelForm):
    class Meta:
        model = SubroutineActiveGlobalVars
        fields = ["id", "subroutine", "variable_name"]

