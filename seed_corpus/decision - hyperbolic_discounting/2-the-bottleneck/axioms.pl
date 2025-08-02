% --------------------
% Axioms: SE best practices (general rules)
% --------------------

% Identify quick‐fix options
quick_fix(Opt) :-
    fix_detail(Opt, _).

% 1) Prefer sustainable solutions with non-high debt risk for long-term reliability
follow_best_practice(User, choose(Opt)) :-
    sustainable_solution(Opt),
    technical_debt_risk(Opt, Risk),
    Risk \= high.

% 2) When the root cause is systemic, favor sustainable solutions
follow_best_practice(User, choose(Opt)) :-
    root_cause(User, bottleneck(_)),
    sustainable_solution(Opt).

% 3) Allow quick fixes only if debt risk is low and performance gain is non-low
follow_best_practice(User, choose(Opt)) :-
    quick_fix(Opt),
    technical_debt_risk(Opt, low),
    performance_gain(Opt, Gain),
    Gain \= low.

% 4) Fallback: choose any option without high technical debt
follow_best_practice(User, choose(Opt)) :-
    technical_debt_risk(Opt, Risk),
    Risk \= high,
    \+ (follow_best_practice(User, choose(_)), !).