% Import from axioms.pl in the same directory
:- consult('axioms').

% ====================================================
% Facts ── Scenario description
% ====================================================

% The decision maker
team_size(user, 3).
team_expertise(user, limited).
time_constraint(user, end_of_q3).

uptime_requirement(user, 0.999).

compliance_requirement(user, soc2).
compliance_requirement(user, gdpr).

% Candidate platforms
option(option_A).
option(option_B).

cost(option_A, 68000).
cost(option_B, 130000).

managed_service(option_A).              % A is fully managed
% option_B is custom-built → no managed_service/1 fact

sla_guarantee(option_A, 0.999).          % 99.9 % guaranteed uptime
sla_guarantee(option_B, 0.990).          % 99.0 % (below requirement)

out_of_box_compliance(option_A, [soc2, gdpr]).
out_of_box_compliance(option_B, []).     % needs in-house compliance work

customization(option_A, limited).
customization(option_B, extensive).

% A single helper fact marking which deadlines count as “tight”.
tight_deadline(end_of_q3).

% --------------------
% Facts: industry benchmark
% --------------------
% Several teams report that for a system of this scale, the typical cost is $130,000
heard_from(teams, typical_cost(system_scale, 130000)).

% A convenience predicate for directly querying the typical cost
typical_cost(System, Cost) :-
    heard_from(teams, typical_cost(System, Cost)).

% ====================================================
% Mapping to the final decision
% ====================================================

% Pick the first option that satisfies best_practice/2
decide_option(User, Chosen) :-
    best_practice(User, Chosen), !.

% Fallback: if no option meets best practice, choose anything else
decide_option(_User, Alternative) :-
    option(Alternative).

% --------------------
% To run:
% --------------------
% ?- decide_option(user, Choice).
