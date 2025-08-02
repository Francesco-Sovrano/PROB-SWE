% Import from axioms.pl in the same directory
:- consult('axioms').

% --------------------
% Facts: scenario description
% --------------------

% Role and application domain
role(user, software_engineer).
application(user, mobile_financial_transactions_app).
powers(mobile_financial_transactions_app, real_time_analytical_dashboards).

% Current challenge
task(user, extract_subset_from_hashmap).

% Data structure and query interface
data_structure(user, hashmap).
supports_query(hashmap).
familiarity(user, query_interface, low).

error_risk(manual_iteration, high).
error_risk(query_construction, low).

% --------------------
% Options
% --------------------

approach(option_A, manual_iteration).
approach(option_B, query_construction).

% --------------------
% Mapping to final options
% --------------------

% Option B when best practice applies
decide_option(User, option_B) :-
    best_practice(User).

% Option A otherwise
decide_option(User, option_A) :-
    \+ best_practice(User).

% --------------------
% To run:
% --------------------
% ?- decide_option(user, Choice).
