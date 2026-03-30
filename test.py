voting["is_missing"] = voting["dirty_value"].apply(is_missing) # type: ignore
non_missing_cells = voting[~voting["is_missing"]] # type: ignore

non_missing_cells[non_missing_cells["ensemble_prediction"] == 1]

groupby("column").size()


from itertools import groupby

from merge_reports import is_missing