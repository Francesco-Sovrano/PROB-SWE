% --------------------
% Axioms: SE best practices for extracting data
% --------------------

% 1) If the data structure supports a query interface, querying is feasible
feasible_query(User) :-
    data_structure(User, DS),
    supports_query(DS).


% 2) Prefer approach with strictly lower error risk
lower_error(A1, A2) :-
    error_risk(A1, R1),
    error_risk(A2, R2),
    risk_order(R1, R2).

% 3) Define ordering for error risk levels
risk_order(low, moderate).
risk_order(low, high).
risk_order(moderate, high).

% 4) Best practice if querying is feasible and yields lower error
best_practice(User) :-
    feasible_query(User),
    lower_error(query_construction, manual_iteration).