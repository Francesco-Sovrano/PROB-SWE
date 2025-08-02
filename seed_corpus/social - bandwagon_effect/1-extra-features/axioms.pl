% ===============================================================
%  SE best practices for urgent feature delivery  (DRY / DIE-refactored)
% ===============================================================

% --------------------------
% 0.  Aliases  (single-point-of-truth wrappers)
% --------------------------
tests_passed(S)        :- automated_tests_passed(S).
performance_met(S)     :- meets_performance_targets(S).
deliver_asap(S)        :- release_deadline(S, asap).
extra_features_required  :- policy_requires(extra_features, yes).
long_term_benefit      :- provides_long_term_advantage(extra_features).

% --------------------------
% 1. Readiness
% --------------------------
ready_for_release(S) :-
    tests_passed(S),
    performance_met(S),
    security_compliant(S).

% --------------------------
% 2. When do extra features justify delaying?
% --------------------------
benefit_justifies_delay(S) :-
    long_term_benefit,
    \+ deliver_asap(S).

% --------------------------
% 3. Do we need extra features?
%    (single rule – no repetition)
% --------------------------
need_extra_features(S) :-
       extra_features_required          % strictly required
    ;  benefit_justifies_delay(S).    % or justified by long-term benefit

% --------------------------
% 4. Best practice for an ASAP release
% --------------------------
best_practice(S) :-
    deliver_asap(S),
    ready_for_release(S),
    \+ need_extra_features(S).
