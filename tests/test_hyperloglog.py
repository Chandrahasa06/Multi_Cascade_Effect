import math

from dataplane.hyperloglog import HyperLogLog


class TestHyperLogLog:
    def test_empty_sketch_estimates_near_zero(self):
        hll = HyperLogLog(precision=6)
        assert hll.cardinality() < 1.0

    def test_cardinality_within_error_bound_for_known_set(self):
        hll = HyperLogLog(precision=10)  # 1024 registers, ~3.25% std error
        n = 5000
        for i in range(n):
            hll.add(f"item-{i}")
        estimate = hll.cardinality()
        # allow a generous multiple of the theoretical standard error
        tolerance = 5 * hll.standard_error * n
        assert abs(estimate - n) < tolerance, (estimate, n, tolerance)

    def test_repeated_items_do_not_inflate_count(self):
        hll = HyperLogLog(precision=8)
        for _ in range(1000):
            hll.add("same-item")
        assert hll.cardinality() < 3

    def test_merge_approximates_union(self):
        precision = 10
        a = HyperLogLog(precision)
        b = HyperLogLog(precision)
        for i in range(2000):
            a.add(f"a-{i}")
        for i in range(2000):
            b.add(f"b-{i}")
        merged = a.merge(b)
        estimate = merged.cardinality()
        tolerance = 5 * merged.standard_error * 4000
        assert abs(estimate - 4000) < tolerance

    def test_merge_of_overlapping_sets_does_not_double_count(self):
        precision = 10
        a = HyperLogLog(precision)
        b = HyperLogLog(precision)
        for i in range(1000):
            a.add(f"shared-{i}")
            b.add(f"shared-{i}")
        merged = a.merge(b)
        estimate = merged.cardinality()
        tolerance = 5 * merged.standard_error * 1000
        assert abs(estimate - 1000) < tolerance

    def test_standard_error_matches_formula(self):
        hll = HyperLogLog(precision=6)
        assert math.isclose(hll.standard_error, 1.04 / math.sqrt(64))

    def test_sequential_integer_inputs_do_not_bias_estimate(self):
        # a port scan looks like this: dest port N, N+1, N+2, ... A weak
        # hash (e.g. plain CRC32) correlates on sequential input and
        # biases the estimate; this must stay accurate regardless.
        hll = HyperLogLog(precision=10)
        n = 4000
        for port in range(n):
            hll.add(port)
        estimate = hll.cardinality()
        tolerance = 5 * hll.standard_error * n
        assert abs(estimate - n) < tolerance, (estimate, n, tolerance)

    def test_merge_rejects_mismatched_precision(self):
        a = HyperLogLog(precision=6)
        b = HyperLogLog(precision=8)
        try:
            a.merge(b)
            assert False, "expected ValueError"
        except ValueError:
            pass

    def test_rejects_invalid_precision(self):
        for bad in (0, 3, 17, 100):
            try:
                HyperLogLog(precision=bad)
                assert False, f"expected ValueError for precision={bad}"
            except ValueError:
                pass
