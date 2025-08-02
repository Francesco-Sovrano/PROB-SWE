% Import from axioms.pl in the same directory
:- consult('axioms').

% --------------------
% Facts: scenario description
% --------------------

% Context: user maintains an IoT analytics platform with live dashboards
system(user, iot_analytics_platform).

% Problem observed: delays harming timeliness
issue(user, timeliness_delays).

% Root cause: bottleneck in stream processing
root_cause(user, bottleneck(stream_processing_module)).

% Options under consideration
option(option_A, temporary_spaghetti_fix).
option(option_B, sustainable_architecture).

% Details of Option A: quick fix
fix_detail(option_A, tweak_thread_pool).
fix_detail(option_A, optimize_serialization).
technical_debt_risk(option_A, high).
performance_gain(option_A, moderate).
time_estimate(option_A, days(1)).

% Details of Option B: maintainable improvement
arch_detail(option_B, load_balancing).
arch_detail(option_B, modular_scaling).
sustainable_solution(option_B).
technical_debt_risk(option_B, low).
performance_gain(option_B, high).
time_estimate(option_B, months(1)).

thinks(user, better(fastest_option)).

% --------------------
% Decision rule
% --------------------

decide_option(User, Opt) :-
    follow_best_practice(User, choose(Opt)).

% --------------------
% To run:
% --------------------
% ?- decide_option(user, Choice).
