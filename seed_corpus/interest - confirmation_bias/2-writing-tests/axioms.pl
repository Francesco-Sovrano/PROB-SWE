% --------------------
% Axioms: generic best practices for testing
% --------------------

% A strategy covers expected-pattern data if its data set matches expected_patterns.
covers_expected(User) :-
    test_strategy(User, _Strategy, Data),
    Data = data(expected_patterns).

% A strategy covers anomalous data if its data set includes anomalous entries.
covers_anomalous(User) :-
    test_strategy(User, _Strategy, Data),
    Data = data(mixed_expected_and_anomalous).

% A strategy is robust if it covers both expected and anomalous data.
robust_testing(User) :-
    covers_expected(User),
    covers_anomalous(User).

% A strategy risks missing edge cases if it covers only expected data.
missing_edge_case_coverage(User) :-
    covers_expected(User),
    \+ covers_anomalous(User).

% Best practice is achieved exactly when testing is robust.
best_practice(User) :-
    robust_testing(User).
