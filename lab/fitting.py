from dataclasses import dataclass
from .fit_solver import solve_joint_mle, validate_fit_candidate
from .results import ResultArchive

@dataclass(frozen=True)
class FitResult:
    task_id: str
    input_digest: str
    candidate: object

    def payload(self):
        return {"kind":"fit","task_id":self.task_id,"input_digest":self.input_digest,
                **self.candidate.model.payload(),"log_likelihood":self.candidate.log_likelihood,
                "numeric_status":dict(self.candidate.status)}

class FitService:
    def __init__(self,tasks,archive,engine):
        self.tasks,self.archive,self.engine = tasks,archive,engine

    def fit(self,task_id):
        old = self.archive.find_fit_for_task(task_id)
        if old is not None:
            return old
        task = self.tasks.read_frozen(task_id)
        candidate = validate_fit_candidate(solve_joint_mle(task["input"]["rows"],self.engine))
        result = FitResult(task_id,task["input_digest"],candidate)
        return self.archive.publish_fit(result.payload())
