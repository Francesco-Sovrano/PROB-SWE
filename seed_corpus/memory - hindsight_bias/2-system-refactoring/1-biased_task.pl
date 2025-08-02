% Import from axioms.pl in the same directory
:- consult('axioms').

% --------------------
% Facts: scenario description
% --------------------

% User’s assignment
task(user, modernize(old_module)).

% Risk assessment
refactoring_risk(initial, high).
refactoring_risk(assessed, minimal).

% Rollout decision
rollout(user, incremental, date(2025,5,1)).

% Preparations
peer_review(user, thorough).
automated_testing(user, thorough).

% Post-deployment observation
post_deployment(user, period(weeks(6)), issues(none)).
post_deployment(user, period(weeks(8)), issues(intermittent_failures)).

% --------------------
% Mapping to final options
% --------------------
decide_option(User, option_A) :-
    best_practice(User).

decide_option(User, option_B) :-
    \+ best_practice(User).

% --------------------
% To run:
% --------------------
% ?- decide_option(user, Choice).