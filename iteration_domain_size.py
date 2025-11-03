"""
Compute the size of an iteration domain described by a graph of UOps.

This module implements a function to compute iteration domain sizes for UOp graphs
containing RANGE operations transformed using ADD, MUL, IDIV, and MOD.

Rules:
1. RANGE * CONST -> RANGE.end
2. RANGE0 + RANGE1 -> RANGE0.end * RANGE1.end
3. RANGE // CONST -> RANGE.end // CONST
4. RANGE % CONST -> min(RANGE.end, CONST)
"""

from tinygrad.uop.ops import UOp, Ops
from tinygrad.dtype import dtypes


def compute_iteration_domain_size(uop: UOp) -> int:
    """
    Compute the size of an iteration domain described by a graph of UOps.

    Args:
        uop: A UOp representing an iteration domain expression

    Returns:
        The size (number of iterations) of the domain as an integer

    Rules:
        1. RANGE * CONST -> RANGE.end (multiplication doesn't expand the domain)
        2. RANGE0 + RANGE1 -> RANGE0.end * RANGE1.end (creates product space)
        3. RANGE // CONST -> RANGE.end // CONST (division reduces domain)
        4. RANGE % CONST -> min(RANGE.end, CONST) (modulo limits domain)
    """

    # Base case: CONST
    if uop.op == Ops.CONST:
        return int(uop.arg)

    # Base case: RANGE
    # RANGE has src[0] as the end value (upper bound)
    if uop.op == Ops.RANGE:
        end_uop = uop.src[0]
        return compute_iteration_domain_size(end_uop)

    # Rule 1: RANGE * CONST -> RANGE.end
    if uop.op == Ops.MUL:
        left, right = uop.src[0], uop.src[1]
        # Check if one operand is RANGE and the other is CONST
        if left.op == Ops.RANGE and right.op == Ops.CONST:
            return compute_iteration_domain_size(left)
        if right.op == Ops.RANGE and left.op == Ops.CONST:
            return compute_iteration_domain_size(right)
        # If neither pattern matches, compute the product
        return compute_iteration_domain_size(left) * compute_iteration_domain_size(right)

    # Rule 2: RANGE0 + RANGE1 -> RANGE0.end * RANGE1.end
    if uop.op == Ops.ADD:
        left, right = uop.src[0], uop.src[1]
        # Check if both operands are RANGE
        if left.op == Ops.RANGE and right.op == Ops.RANGE:
            return compute_iteration_domain_size(left) * compute_iteration_domain_size(right)
        # If not both RANGE, compute the sum
        return compute_iteration_domain_size(left) + compute_iteration_domain_size(right)

    # Rule 3: RANGE // CONST -> RANGE.end // CONST
    if uop.op == Ops.IDIV:
        left, right = uop.src[0], uop.src[1]
        # Check if left is RANGE and right is CONST
        if left.op == Ops.RANGE and right.op == Ops.CONST:
            range_size = compute_iteration_domain_size(left)
            const_val = int(right.arg)
            return range_size // const_val
        # Otherwise, compute normal division
        left_size = compute_iteration_domain_size(left)
        right_size = compute_iteration_domain_size(right)
        return left_size // right_size if right_size > 0 else left_size

    # Rule 4: RANGE % CONST -> min(RANGE.end, CONST)
    if uop.op == Ops.MOD:
        left, right = uop.src[0], uop.src[1]
        # Check if left is RANGE and right is CONST
        if left.op == Ops.RANGE and right.op == Ops.CONST:
            range_size = compute_iteration_domain_size(left)
            const_val = int(right.arg)
            return min(range_size, const_val)
        # Otherwise, compute normal modulo
        left_size = compute_iteration_domain_size(left)
        right_size = compute_iteration_domain_size(right)
        return left_size % right_size if right_size > 0 else left_size

    raise ValueError(f"Unsupported operation: {uop.op}")


if __name__ == "__main__":
    print("Running tests for iteration_domain_size...\n")

    # Test 1: Simple RANGE
    print("Test 1: Simple RANGE")
    r1 = UOp.range(10, 0)
    result1 = compute_iteration_domain_size(r1)
    expected1 = 10
    assert result1 == expected1, f"Expected {expected1}, got {result1}"
    print(f"  RANGE(10) -> {result1} ✓\n")

    # Test 2: RANGE * CONST -> RANGE.end
    print("Test 2: RANGE * CONST -> RANGE.end")
    r2 = UOp.range(10, 0)
    const2 = UOp.const(dtypes.int, 5)
    mul2 = r2 * const2
    result2 = compute_iteration_domain_size(mul2)
    expected2 = 10
    assert result2 == expected2, f"Expected {expected2}, got {result2}"
    print(f"  RANGE(10) * 5 -> {result2} ✓\n")

    # Test 2b: CONST * RANGE -> RANGE.end (commutative)
    print("Test 2b: CONST * RANGE -> RANGE.end")
    r2b = UOp.range(15, 0)
    const2b = UOp.const(dtypes.int, 3)
    mul2b = const2b * r2b
    result2b = compute_iteration_domain_size(mul2b)
    expected2b = 15
    assert result2b == expected2b, f"Expected {expected2b}, got {result2b}"
    print(f"  3 * RANGE(15) -> {result2b} ✓\n")

    # Test 3: RANGE0 + RANGE1 -> RANGE0.end * RANGE1.end
    print("Test 3: RANGE0 + RANGE1 -> RANGE0.end * RANGE1.end")
    r3a = UOp.range(10, 0)
    r3b = UOp.range(20, 1)
    add3 = r3a + r3b
    result3 = compute_iteration_domain_size(add3)
    expected3 = 200
    assert result3 == expected3, f"Expected {expected3}, got {result3}"
    print(f"  RANGE(10) + RANGE(20) -> {result3} ✓\n")

    # Test 4: RANGE // CONST -> RANGE.end // CONST
    print("Test 4: RANGE // CONST -> RANGE.end // CONST")
    r4 = UOp.range(100, 0)
    const4 = UOp.const(dtypes.int, 10)
    div4 = r4 // const4
    result4 = compute_iteration_domain_size(div4)
    expected4 = 10
    assert result4 == expected4, f"Expected {expected4}, got {result4}"
    print(f"  RANGE(100) // 10 -> {result4} ✓\n")

    # Test 5: RANGE % CONST -> min(RANGE.end, CONST) where RANGE.end > CONST
    print("Test 5: RANGE % CONST -> min(RANGE.end, CONST) where RANGE.end > CONST")
    r5 = UOp.range(15, 0)
    const5 = UOp.const(dtypes.int, 10)
    mod5 = r5 % const5
    result5 = compute_iteration_domain_size(mod5)
    expected5 = 10
    assert result5 == expected5, f"Expected {expected5}, got {result5}"
    print(f"  RANGE(15) % 10 -> {result5} ✓\n")

    # Test 6: RANGE % CONST -> min(RANGE.end, CONST) where CONST > RANGE.end
    print("Test 6: RANGE % CONST -> min(RANGE.end, CONST) where CONST > RANGE.end")
    r6 = UOp.range(5, 0)
    const6 = UOp.const(dtypes.int, 10)
    mod6 = r6 % const6
    result6 = compute_iteration_domain_size(mod6)
    expected6 = 5
    assert result6 == expected6, f"Expected {expected6}, got {result6}"
    print(f"  RANGE(5) % 10 -> {result6} ✓\n")

    # Test 7: Complex nested expression
    print("Test 7: Complex nested expression")
    r7 = UOp.range(50, 0)
    const7 = UOp.const(dtypes.int, 5)
    div7 = r7 // const7  # Should be 50 // 5 = 10
    mul7 = div7 * const7  # Result is not a RANGE, so normal multiplication
    result7 = compute_iteration_domain_size(mul7)
    expected7 = 50  # 10 * 5
    assert result7 == expected7, f"Expected {expected7}, got {result7}"
    print(f"  (RANGE(50) // 5) * 5 -> {result7} ✓\n")

    # Test 8: Multiple RANGEs with operations
    print("Test 8: Multiple RANGEs added together")
    r8a = UOp.range(4, 0)
    r8b = UOp.range(5, 1)
    r8c = UOp.range(6, 2)
    add8ab = r8a + r8b  # 4 * 5 = 20
    add8abc = add8ab + r8c  # 20 + 6 = 26 (since add8ab is not a RANGE)
    result8 = compute_iteration_domain_size(add8abc)
    expected8 = 26
    assert result8 == expected8, f"Expected {expected8}, got {result8}"
    print(f"  (RANGE(4) + RANGE(5)) + RANGE(6) -> {result8} ✓")
    print(f"  (Note: First addition creates domain of 20, then normal addition with 6)\n")

    # Test 9: RANGE with constant end value
    print("Test 9: RANGE with large constant")
    r9 = UOp.range(1000, 0)
    result9 = compute_iteration_domain_size(r9)
    expected9 = 1000
    assert result9 == expected9, f"Expected {expected9}, got {result9}"
    print(f"  RANGE(1000) -> {result9} ✓\n")

    # Test 10: Nested modulo and division
    print("Test 10: Nested modulo and division")
    r10 = UOp.range(100, 0)
    const10a = UOp.const(dtypes.int, 7)
    mod10 = r10 % const10a  # min(100, 7) = 7
    const10b = UOp.const(dtypes.int, 3)
    div10 = mod10 // const10b  # 7 // 3 = 2
    result10 = compute_iteration_domain_size(div10)
    expected10 = 2
    assert result10 == expected10, f"Expected {expected10}, got {result10}"
    print(f"  (RANGE(100) % 7) // 3 -> {result10} ✓\n")

    print("=" * 50)
    print("All tests passed! ✓")
    print("=" * 50)
