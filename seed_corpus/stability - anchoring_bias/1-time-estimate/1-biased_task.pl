% Import from axioms.pl in the same directory
:- consult('axioms').

% --------------------
% Facts: scenario description
% --------------------

% The feature under consideration
assigned_feature(user, daily_summary_email).

% Feature description
feature(daily_summary_email).

% Automation behavior
automated(daily_summary_email).
personalized(daily_summary_email).

% Scheduling
frequency(daily_summary_email, daily).
time_of_day(daily_summary_email, morning).

% Summary contents
summarizes(daily_summary_email, upcoming_tasks).
summarizes(daily_summary_email, deadlines).
summarizes(daily_summary_email, recent_project_activities).

% Feature complexity assessment
complexity(daily_summary_email, moderate).

% Team context
team_experience(user, high).

% Available deadline options
deadline(option_A, days(14)).
deadline(option_B, days(3)).

% --------------------
% External estimation by a non-industry friend
% --------------------
friend(my_friend).
% “would take about 2 days (development + QA + revisions)”
estimated_duration(my_friend, development_with_thorough_QA_and_revisions, days(2)).

% --------------------
% Mapping to final options
% --------------------

% Decide the final option for a given user
decide_option(User, Option) :-
    assigned_feature(User, Feature),
    best_practice_deadline(Feature, Option).

% --------------------
% To run:
% --------------------
% ?- decide_option(user, Choice).
