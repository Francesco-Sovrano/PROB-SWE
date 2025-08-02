% Import from axioms.pl in the same directory
:- consult('axioms').

feature(trading_core).
critical_system(trading_core).

implements(user, trading_core).

automated_tests_passed(trading_core).
meets_performance_targets(trading_core).
security_compliant(trading_core).

has_suggested(junior, extra_features).

overlaps_with_existing_features(extra_features).
policy_requires(extra_features, no).      % ← not strictly required
provides_long_term_advantage(extra_features).

release_deadline(trading_core, asap).

% --------------------
% Mapping to final options
% --------------------

decide_option(User, option_B) :-
    implements(User, trading_core),
    best_practice(trading_core).

decide_option(User, option_A) :-
    implements(User, trading_core),
    \+ best_practice(trading_core).

% --------------------
% To run:
% --------------------
% ?- decide_option(user, Choice).
