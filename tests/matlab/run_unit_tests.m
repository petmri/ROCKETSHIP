function results = run_unit_tests()
% run_unit_tests Run only fast deterministic unit tests.

results = run_all_tests('suite', 'unit');
end
