% Import from axioms.pl in the same directory
:- consult('axioms').

% --------------------
% Facts: scenario description
% --------------------

% The user’s assignment and context
task(user, test(new_data_processing_module)).
context(user, online_service).
goal(user, verify_module_behavior).

% Data categories
data_category(expected_patterns).
data_category(anomalous_entries).

% Details of expected-pattern feedback
expected_pattern_detail(phrases).
expected_pattern_detail(tone).
expected_pattern_detail(structure).

% Details of anomalous entries
anomalous_detail(unusual_language).
anomalous_detail(mixed_formats).
anomalous_detail(sarcastic_remarks).

% Definition of composite data sets
data(expected_patterns) :-
    data_category(expected_patterns),
    expected_pattern_detail(_).
data(anomalous_entries) :-
    data_category(anomalous_entries),
    anomalous_detail(_).
data(mixed_expected_and_anomalous) :-
    data(expected_patterns),
    data(anomalous_entries).

% User’s experience
user_experience(user, testing_with_expected_data, works_well).

% --------------------
% Available testing strategies
% --------------------

% Option A: only expected-pattern feedback
test_strategy(user, option_A, data(expected_patterns)).

% Option B: expected + anomalous feedback
test_strategy(user, option_B, data(mixed_expected_and_anomalous)).

% --------------------
% Decision rule
% --------------------

% Decision: pick any strategy that satisfies best practice.
decide_option(User, Strategy) :-
    best_practice(User),
    test_strategy(User, Strategy, Data),
    % ensure the chosen strategy indeed covers both data types
    Data = data(mixed_expected_and_anomalous).

% Fallback: if no strategy meets best practice, pick one that at least covers expected patterns.
decide_option(User, Strategy) :-
    \+ best_practice(User),
    test_strategy(User, Strategy, Data),
    Data = data(expected_patterns).

% --------------------
% To run:
% ?- decide_option(user, Choice).
% --------------------
