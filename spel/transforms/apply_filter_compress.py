from spel.scripts.analysis_cache import load_analysis
from spel.transforms.refactor_tools import compress_on_filter
from pprint import pprint

fut = load_analysis(False)

lakefluxes = fut.subroutine_dict['lakefluxesmod::lakefluxes']

# print( lakefluxes.local_variables )
# pprint(lakefluxes.record.calls)
pprint(lakefluxes.record.loops)

# compress_on_filter(lakefluxes)
