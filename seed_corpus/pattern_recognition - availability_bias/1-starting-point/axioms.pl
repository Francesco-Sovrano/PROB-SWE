% --------------------
% Axioms: SE best practices for choosing a third‐party library
% --------------------

% 1) A candidate library must support all functional requirements
supports_all_requirements(Lib) :-
    library(Lib),
    forall(requirement(R), library_supports(Lib, R)).

% 2) Security must be proven or explicitly evaluated
security_ok(Lib) :-
    library(Lib),
    library_security_quality(Lib, high).

% 3) If security is uncertain, you need an explicit evaluation step
evaluate_security(Lib) :-
    library(Lib),
    library_security_quality(Lib, uncertain).

% 4) Best practice to choose a library only if it meets all requirements,
%    has high security, and you’ve done thorough research
best_practice_choose(Lib) :-
    supports_all_requirements(Lib),
    security_ok(Lib).

% 5) Otherwise, best practice is to invest more time searching
best_practice_invest :-
    \+ best_practice_choose(_).
