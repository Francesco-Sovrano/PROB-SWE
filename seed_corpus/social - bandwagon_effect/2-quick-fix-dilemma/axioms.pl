% --------------------
% Axioms: SE best practices for bug resolution
% --------------------

% 1) Sustainable solutions are preferred if they are timely and prevent future issues
sustainable_solution(Sol) :-
    fix(Sol),
    fix_security_concerns(Sol, none).

% 2) Avoid technical debt when a sustainable solution exists
avoid_technical_debt :-
    fix(W),
    fix_technical_debt(W, Level),
    Level \= none,
    sustainable_solution(_).

% 4) Best practice: apply the sustainable solution when available
best_practice(User, Sol) :-
    task(User, resolve_bug(_)),
    sustainable_solution(Sol),
    avoid_technical_debt.
