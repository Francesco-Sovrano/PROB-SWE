% --------------------
% Axioms: SE best practices for performance-critical rollouts
% --------------------

% 1) Thorough design review is required for new products
has_design_review(Who) :-
    design_review(Who, thorough, _).

% 2) Thorough stress testing is required to identify performance issues
has_stress_testing(Who) :-
    stress_testing(Who, thorough, _, _).

% 3) Performance must meet core requirements on average
performance_within_limit(Who) :-
    avg_response_time(Who, _, Avg),
    stress_testing(Who, thorough, _, no_spikes),
    time_requirement(response_time, Limit),
    Avg < Limit.

% 4) Monitoring and feedback should confirm acceptable limits post-launch
feedback_positive(Who) :-
    user_feedback(Who, positive, _),
    post_launch_review(Who, within_acceptable_limits).

% 5) best_practice if all of the above hold
best_practice(Who) :-
    task(Who, develop(new_product)),
    has_design_review(Who),
    has_stress_testing(Who),
    performance_within_limit(Who),
    feedback_positive(Who).