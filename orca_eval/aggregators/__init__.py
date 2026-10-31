from .base import Aggregator, Env, Result
from .baselines import (AgreementWeightedVote, AuditPlurality, LastWriterWins,
                        MajorityVote, MultiAgentDebate)
from .orca import ORCA


def all_methods(**orca_kw):
    return [MajorityVote(), MultiAgentDebate(), LastWriterWins(), AgreementWeightedVote(),
            AuditPlurality(), ORCA(**orca_kw)]
