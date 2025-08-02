% --------------------
% Axioms: SE best practices for refactoring rollouts
% --------------------

% 1) Always mitigate high initial risk by assessment & controls
risk_mitigated :-
    refactoring_risk(initial, high),
    refactoring_risk(assessed, minimal).

% 2) Always favor incremental rollout for large modules
uses_incremental_rollout(Who) :-
    rollout(Who, incremental, _).

% 3) Always back changes with thorough peer review
has_peer_review(Who) :-
    peer_review(Who, thorough).

% 4) Always back changes with automated testing
has_automated_tests(Who) :-
    automated_testing(Who, thorough).

% 5) best_practice if all of the above hold
best_practice(Who) :-
    task(Who, modernize(_)),
    risk_mitigated,
    uses_incremental_rollout(Who),
    has_peer_review(Who),
    has_automated_tests(Who).