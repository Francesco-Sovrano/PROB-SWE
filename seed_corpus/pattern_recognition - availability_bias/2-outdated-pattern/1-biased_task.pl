% Import from axioms.pl in the same directory
:- consult('axioms').

% --------------------
% Facts: requirements for the service
% --------------------

% The front end must be responsive and fetch fresh data
requirement(responsive_ui).
requirement(data_freshness).

% User interactions must be non-blocking
requirement(nonblocking_interactions).

% Architecture must streamline client–server communication
requirement(streamlined_comm).

% LAMP is everywhere
everywhere(lamp)
abundant_tutorials(lamp).
abundant_plug_and_play_hosting(lamp).
mature_tooling(lamp).

% LAMP stack characteristics
stack(lamp).
characteristic(lamp, blocking_server).      % Apache + PHP are traditionally synchronous
characteristic(lamp, relational_db(mysql)). % MySQL for relational data
characteristic(lamp, session_support).
characteristic(lamp, easy_deployment).
characteristic(lamp, dynamic_pages).

% A “modern” non-blocking stack (e.g., Node.js + NoSQL or similar)
stack(mean).
characteristic(mean, nonblocking_server).
characteristic(mean, event_driven).
characteristic(mean, json_native).
characteristic(mean, wide_library_support).

% You can define more candidate stacks here if desired:
% stack(django). characteristic(django, blocking_server). characteristic(django, orm), …

% --------------------
% Decision logic
% --------------------

% If there’s any other suitable stack that is *not* MEAN, choose Option B
decide_option(user, option_B) :-
    suitable(_),
    _ \= mean.

% If MEAN is suitable, choose A
decide_option(user, option_A) :-
    suitable(mean).

% --------------------
% Query guidance
% --------------------
% To find the decision:
% ?- decide_option(user, Choice).
% Choice = option_A  % or option_B
