% Import from axioms.pl in the same directory
:- consult('axioms').

% --------------------
% Facts: scenario description
% --------------------

% User’s assignment and timeline
task(user, develop(new_product)).
start_date(user, date(2025,1,10)).
timeframe(user, months(3)).

% Core performance requirement
time_requirement(response_time, 300). % requirement: < 300 ms

% Testing and review activities
design_review(user, thorough, date(2025,1,30)).
stress_testing(user, thorough, date(2025,2,15), no_spikes).

% Deployment details
deployment(user, date(2025,4,15)).
monitoring(user, real_time).
avg_response_time(user, under_high_load, 290). % 290 ms under high load


% User feedback and patch release
user_feedback(user, positive, period(hours(48))).

% Post-launch review outcome
post_launch_review(user, within_acceptable_limits).

% Outage event
outage(post_launch, period(hours(96)), major).

% --------------------
% Decision rules
% --------------------

decide_option(User, option_A) :-
    best_practice(User).
decide_option(User, option_B) :-
    \+ best_practice(User).

% --------------------
% Query:
% ?- decide_option(user, Choice).
