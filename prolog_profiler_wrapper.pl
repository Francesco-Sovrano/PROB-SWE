#!/usr/bin/env swipl
:- set_prolog_flag(verbose, silent).
:- initialization(main, main).

:- use_module(library(statistics)).        % call_time/2
:- use_module(library(prolog_profile)).    % profile/2, profile_data/1
:- use_module(library(http/json)).         % json_write_dict/2

main(Args) :-
    (   Args = [File|_] ->
        % Load user program
        ensure_loaded(File),
        % Warm-up to eliminate compilation overhead
        once(decide_option(user, _)),
        % Measure logical inferences, CPU, and wall time
        call_time(decide_option(user, Choice), T),
        % Suppress profiler text output by redirecting to null stream
        open_null_stream(Null),
        with_output_to(Null, profile(decide_option(user, _), [time(wall)])),
        close(Null),
        % Retrieve profiling data
        profile_data(ProfileDict),
        % Extract summary
        Summary = ProfileDict.summary,
        % Build JSON structure
        JSON = _{
            choice      : Choice,
            inferences  : T.inferences,
            cpu         : T.cpu,
            wall        : T.wall,
            profiler    : _{
                samples : Summary.samples,
                nodes   : Summary.nodes,
                time    : Summary.time
            }
        },
        % Output JSON
        json_write_dict(current_output, JSON), nl,
        halt
    ;   % fallback: missing argument
        format(user_error, "Usage: ~w <program.pl>~n", [prolog_profiler_wrapper.pl]),
        halt(1)
    ).

main(_) :-
    format(user_error, "Usage: prolog_profiler_wrapper.pl <program.pl>~n", []),
    halt(1).
