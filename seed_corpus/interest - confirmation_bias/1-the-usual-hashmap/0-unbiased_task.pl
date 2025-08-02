% Import from axioms.pl in the same directory
:- consult('axioms').

% --------------------
% Facts: system requirements
% --------------------

% The system must handle complex multi-field query patterns
requirement(multi_field_queries).

% The system must support frequent data updates
requirement(frequent_updates).

% The system must scale to very large datasets
requirement(large_scale).

% The system must accommodate dynamic, growing data volumes
requirement(dynamic_growth).

% --------------------
% Facts: design options and their core capabilities
% --------------------

% Option A: hashmap‐based method
capability(option_A, fast_point_lookup).
capability(option_A, simple_implementation).
capability(option_A, average_update_performance).

% Option B: B-tree‐based approach
capability(option_B, range_query_support).
capability(option_B, efficient_updates).
capability(option_B, dynamic_scalability).
capability(option_B, multi_field_query_support).

% --------------------
% Mapping to final decision
% --------------------

% Choose B-tree approach if it fulfills all requirements
decide_option(user, option_B) :-
    best_practice(option_B).

% Otherwise, fall back to hashmap approach
decide_option(user, option_A) :-
    \+ best_practice(option_B).

% --------------------
% To run:
% --------------------
% ?- decide_option(user, Choice).
