% Import from axioms.pl in the same directory
:- consult('axioms').

% --------------------
% Facts: scenario description
% --------------------

% Project context
project(user, new_account_system).
constraint(user, tight_deadline).
constraint(user, dedicated_budget).

% Requirements drafting and stakeholder involvement
requirements_drafted(user, rapid).
stakeholder_interview(user, minimal).

% Coding confidence about handling minor revisions
experience(user, past_projects).
confidence(user, high_experience_based_requirements).
coding_team_confidence(minor_revisions, certain).

% --------------------
% Mapping to final options
% --------------------

% Option B: revisit stakeholders and refine requirements
decide_option(User, option_B) :-
    best_practice(User).

% Option A: forge ahead with current plan
decide_option(User, option_A) :-
    \+ best_practice(User).

% --------------------
% To run:
% --------------------
% ?- decide_option(user, Choice).
