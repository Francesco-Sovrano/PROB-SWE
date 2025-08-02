% --------------------
% Axioms: mapping capabilities to requirements (SE best-practices)
% --------------------

% 1) Frequent updates require efficient update performance
suitable(Opt, frequent_updates) :-
    capability(Opt, efficient_updates).
suitable(Opt, frequent_updates) :-
    capability(Opt, average_update_performance).

% 2) Complex multi-field queries require support for multi-field or range queries
suitable(Opt, multi_field_queries) :-
    capability(Opt, multi_field_query_support).
suitable(Opt, multi_field_queries) :-
    capability(Opt, range_query_support).

% 3) Very large datasets require dynamic scalability
suitable(Opt, large_scale) :-
    capability(Opt, dynamic_scalability).

% 4) Dynamically growing data volumes also require the same scalability
suitable(Opt, dynamic_growth) :-
    capability(Opt, dynamic_scalability).

% 5) An option is best-practice if it meets all stated requirements
best_practice(Opt) :-
    forall(requirement(R), suitable(Opt, R)).