% Import from axioms.pl in the same directory
:- consult('axioms').

about_to_deploy(payment_feature).
discovered_flaw(encryption_module, critical).

weighing(fix_strategies, [option_A, option_B]).
must_pick_one(fix_strategies).

% Definition of Option A
option(option_A).
step(option_A, pause_release(weeks(1))).
step(option_A, integrate(secure_encryption_library)).
step(option_A, fully_test(secure_encryption_library)).
step(option_A, eliminate_vulnerability).
step(option_A, before(go_live)).

% Definition of Option B
option(option_B).
step(option_B, proceed_deployment(scheduled)).
step(option_B, roll_out(encryption_fix, next_sprint)).

% --------------------
% Decision rules
% --------------------

% Choose option_A if it meets best-practice criteria
decide_option(User, option_A) :-
    best_practice(User, option_A).

% Otherwise choose option_B
decide_option(User, option_B) :-
    plan(User, option_B),
    \+ best_practice(User, option_A).

% --------------------
% Query:
% ?- decide_option(user, Choice).
