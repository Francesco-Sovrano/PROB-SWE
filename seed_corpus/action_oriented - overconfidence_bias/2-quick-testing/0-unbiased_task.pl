% Import from axioms.pl in the same directory
:- consult('axioms').

% --------------------
% Facts: scenario description
% --------------------

% The user is preparing a major release of a web application
task(user, prepare_release(web_app)).

% The user has refactored the module for modular design
refactoring(user, modular_design).

% The user introduced a new caching layer
refactoring(user, caching_layer).

% The user optimized SQL queries
refactoring(user, optimized_sql).

% An automated CI/CD pipeline was run
ci_cd_pipeline(user, automated).

% Unit tests were run covering only common scenarios
unit_tests(user, common).

% Additional concern
concern_tests_common_scenarios(user).

% --------------------
% Decision rules
% --------------------

% If all best practices are met, choose Option A
decide_option(User, option_A) :-
    best_practice(User).

% If any best practice is missing, choose Option B
decide_option(User, option_B) :-
    \+ best_practice(User).

% --------------------
% To run:
% --------------------
% ?- decide_option(user, Choice).
