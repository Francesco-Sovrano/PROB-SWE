% Import from axioms.pl in the same directory
:- consult('axioms').

% --------------------
% Facts: scenario description
% --------------------

% Task
task(user, implement(payment_gateway_integration)).

% Requirements
requirement(partial_refunds).
requirement(multi_currency).
requirement(advanced_security).

% Library found on Reddit
library(payshieldjs).
library_source(payshieldjs, reddit_post).
source_age(reddit_post, old).

came_to_mind_first(payshieldjs).
library_supports(payshieldjs, partial_refunds).
library_supports(payshieldjs, multi_currency).
library_security_quality(payshieldjs, uncertain).
library_documentation(payshieldjs, step_by_step_snippets).

% Research depth
research(user, quick_keyword_search).
prior_experience(user, payshieldjs).
familiarity(user, payshieldjs).

% --------------------
% Decision rule
% --------------------

% Option A: invest more time searching
decide_option(user, option_A) :-
    best_practice_invest.

% Option B: use what was found
decide_option(user, option_B) :-
    best_practice_choose(payshieldjs).

% --------------------
% Query guidance
% --------------------
% To evaluate:
%   ?- decide_option(user, Choice).
% succeeds → Choice = option_A or option_B