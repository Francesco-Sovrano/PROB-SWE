:- discontiguous stack/1.
:- discontiguous characteristic/2.

% --------------------
% Axioms: SE best practices for web-service stacks
% --------------------

% 1) A stack “meets” the responsive UI requirement if it supports dynamic pages or JSON APIs
meets(responsive_ui, Stack) :-
    characteristic(Stack, dynamic_pages).
meets(responsive_ui, Stack) :-
    characteristic(Stack, json_native).

% 2) A stack “meets” data-freshness if it can serve fresh data (i.e. non-caching by default or easy cache-control)
meets(data_freshness, Stack) :-
    characteristic(Stack, json_native).
meets(data_freshness, Stack) :-
    characteristic(Stack, session_support).

% 3) A stack “meets” non-blocking interactions if its server is non-blocking
meets(nonblocking_interactions, Stack) :-
    characteristic(Stack, nonblocking_server).

% 4) A stack “meets” streamlined communication if it is event-driven or JSON-native
meets(streamlined_comm, Stack) :-
    characteristic(Stack, event_driven).
meets(streamlined_comm, Stack) :-
    characteristic(Stack, json_native).

% 5) A stack is “suitable” if it meets *all* requirements
suitable(Stack) :-
    stack(Stack),
    \+ ( requirement(Req),
         \+ meets(Req, Stack)
       ).