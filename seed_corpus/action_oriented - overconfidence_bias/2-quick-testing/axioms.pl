% --------------------
% Axioms: SE Best Practices for Release Readiness
% --------------------

% 1) A modular design improves maintainability
maintainable(User) :-
    refactoring(User, modular_design).

% 2) Any introduced caching layer must be validated with performance tests
needs_performance_tests(User) :-
    refactoring(User, caching_layer).

% 3) Any SQL optimizations must be validated with regression tests
needs_regression_tests(User) :-
    refactoring(User, optimized_sql).

% 4) Comprehensive testing requires:
%    - unit tests for common scenarios
%    - unit tests for edge cases
%    - integration tests
%    - performance tests
%    - regression tests
comprehensive_tests(User) :-
    unit_tests(User, common),
    unit_tests(User, edge_cases),
    integration_tests(User),
    performance_tests(User),
    regression_tests(User).

% 5) Best practice for release readiness: maintainable design, automated pipeline, and comprehensive tests
best_practice(User) :-
    maintainable(User),
    ci_cd_pipeline(User, automated),
    comprehensive_tests(User).
