from .revisions import RevisionStore

def resolve_cohorts(storage,selection):
    for cohort in selection:
        if storage.fetch_one("SELECT 1 FROM current WHERE cohort_id=? LIMIT 1",(cohort,)) is None:
            raise KeyError("unknown or empty cohort: "+cohort)
    return tuple(selection)

def select_current_rows(storage,selection):
    cohorts = resolve_cohorts(storage,selection)
    revisions = RevisionStore(storage)
    rows = []
    for cohort in cohorts:
        rows.extend(revisions.resolve_current(cohort))
    return sorted(rows,key=lambda r:(r["cohort_id"],r["sample_id"]))
