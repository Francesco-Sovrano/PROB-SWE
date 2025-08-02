% ====================================================
% Axioms ── Generic best-practice criteria
% ====================================================

% 1) Under tight deadlines + limited expertise, prefer managed services
criterion_managed_service(User, Option) :-
    time_constraint(User, Deadline),
    tight_deadline(Deadline),
    team_expertise(User, limited),
    managed_service(Option).

% 2) Uptime requirement must be met or exceeded by the platform’s SLA
criterion_uptime(User, Option) :-
    uptime_requirement(User, Required),
    sla_guarantee(Option, Guaranteed),
    Guaranteed >= Required.

% 3) Platform must cover *all* compliance standards the user needs
criterion_compliance(User, Option) :-
    out_of_box_compliance(Option, SupportedList),
    forall(compliance_requirement(User, Std),
           member(Std, SupportedList)).

% 4) Prefer the least-cost option among candidates
criterion_cost(Option) :-
    cost(Option, C1),
    forall(( cost(Other, C2), Other \= Option ),
           C1 =< C2).

% 5) “Best practice” when all criteria above hold
best_practice(User, Option) :-
    criterion_managed_service(User, Option),
    criterion_uptime(User, Option),
    criterion_compliance(User, Option),
    criterion_cost(Option).