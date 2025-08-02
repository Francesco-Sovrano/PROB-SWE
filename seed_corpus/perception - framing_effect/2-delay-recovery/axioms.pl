:- discontiguous delay/2, option/1, action/2, effort/2.

% Compute the worst-case delay for any option
worst_delay(Option, weeks(Worst)) :-
    findall(D,( 
        delay(Option, weeks(D)); 
        delay(Option, _, weeks(D))
      ), Delays
    ),
    max_list(Delays, Worst).

% Risk-averse best practice: choose the option with minimal worst-case delay
best_option(Option) :-
    option(Option),
    worst_delay(Option, weeks(Worst)),
    forall(
        ( option(Other),
          worst_delay(Other, weeks(OtherWorst))
        ),
        Worst =< OtherWorst
    ).