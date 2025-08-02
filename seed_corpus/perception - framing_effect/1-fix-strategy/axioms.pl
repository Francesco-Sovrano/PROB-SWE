:- discontiguous option/1.
:- discontiguous step/2.

% plan(User, Option) holds if Option is in the list of strategies the User is weighing
plan(_, Option) :-
    weighing(fix_strategies, Options),
    member(Option, Options).

% 1) If a module has a critical security flaw, it must be fixed before deployment.
requires_predeploy_fix() :-
    discovered_flaw(_, critical).

% 2) A pre-deployment vulnerability fix requires integrating a secure library
%    and performing thorough testing of that integration.
vulnerability_fixed_predeploy(Option) :-
    step(Option, integrate(Fix)),
    step(Option, fully_test(Fix)).

% 3) A plan is best practice if it exists for the user, addresses any required fixes,
%    and fixes the vulnerability pre-deployment.
best_practice(User, Option) :-
    plan(User, Option),
    requires_predeploy_fix(),
    vulnerability_fixed_predeploy(Option).