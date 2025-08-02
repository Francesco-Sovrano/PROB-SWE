% --------------------
% Axioms: SE best practices for requirements gathering
% --------------------

% Thorough stakeholder interviews produce complete requirements
complete_requirements(User) :-
    stakeholder_interview(User, thorough).

% Incomplete requirements (lack of thorough interviews) introduce rework risk
risk_rework(User) :-
    \+ complete_requirements(User).

% Tight launch deadlines amplify the impact of rework risk
high_rework_risk(User) :-
    constraint(User, tight_deadline),
    risk_rework(User).

% When high rework risk exists under a tight deadline, best practice is to refine requirements via detailed interviews
should_refine_requirements(User) :-
    high_rework_risk(User).

% Best practice if requirements are complete or if refinement is needed
best_practice(User) :-
    complete_requirements(User).
best_practice(User) :-
    should_refine_requirements(User).
