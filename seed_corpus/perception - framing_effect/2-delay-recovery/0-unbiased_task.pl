:- consult('axioms').

slip(mobile_app_launch, weeks(4)).
cause(mobile_app_launch, unexpected_integration_work).
weighing(recovery_strategies, [option_A, option_B]).
must_pick_one(recovery_strategies).

option(option_A).
action(option_A, disable_optional_onboarding_flow).
effort(option_A, half_day).
outcome(option_A, regain(weeks(2))).
delay(option_A, weeks(2)).

option(option_B).
action(option_B, introduce_build_automation_script).
effort(option_B, weeks(1)).
outcome(option_B, success, regain(weeks(4))).
outcome(option_B, failure, regain(weeks(0))).
delay(option_B, success, weeks(0)).
delay(option_B, failure, weeks(5)).

% --------------------
% Decision rules
% --------------------

% If an option meets the risk-averse best practice, it is the recommended choice
decide_option(user, option_A) :-
    best_option(option_A).

decide_option(user, option_B) :-
    best_option(option_B).

% --------------------
% Query:
% ?- decide_option(user, Choice).
