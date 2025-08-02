% Import from axioms.pl in the same directory
:- consult('axioms').

% --------------------
% Facts: scenario description
% --------------------

% User’s assignment
task(user, resolve_bug(payment_processing_system)).

% Bug characteristics
bug_severity(payment_processing_system, high).
bug_effects(payment_processing_system, intermittent_delays_under_heavy_load).

% Quick workaround details
fix(patch_A).
fix_time(patch_A, days(2)).
fix_technical_debt(patch_A, high).
fix_security_concerns(patch_A, unknown).

% Robust fix details
fix(patch_B).
fix_time(patch_B, days(3)).
fix_technical_debt(patch_B, none).
fix_security_concerns(patch_B, none).

heard_that(several_teams_using(payment_processing_system), chose(patch_a)).

% --------------------
% Mapping to final options
% --------------------

decide_option(User, option_B) :-
    best_practice(User, patch_B).

decide_option(User, option_A) :-
    \+ best_practice(User, _).

% --------------------
% To run:
% --------------------
% ?- decide_option(user, Choice).
