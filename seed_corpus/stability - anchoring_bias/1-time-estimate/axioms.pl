% --------------------
% Axioms: SE best practices for feature deadlines (generic)
% --------------------

% Define threshold for short deadline (in days)
short_deadline_threshold(10).

% Complexity categories for features
is_low_complexity(Feature) :- complexity(Feature, low).
is_moderate_complexity(Feature) :- complexity(Feature, moderate).
is_high_complexity(Feature) :- complexity(Feature, high).

% Short deadlines (< threshold) are appropriate only for low-complexity features
short_deadline(Feature) :-
    is_low_complexity(Feature).

% Moderate or high complexity features require longer deadlines
requires_long_deadline(Feature) :-
    ( is_moderate_complexity(Feature)
    ; is_high_complexity(Feature)
    ).

% Best practice: select an option whose deadline matches complexity requirements
best_practice_deadline(Feature, Option) :-
    deadline(Option, days(Days)),
    short_deadline_threshold(Threshold),
    (
        % Case 1: Feature is low complexity and Option is a short deadline
        short_deadline(Feature),
        Days < Threshold
    ;
        % Case 2: Feature is moderate/high complexity and Option is a long deadline
        requires_long_deadline(Feature),
        Days >= Threshold
    ).