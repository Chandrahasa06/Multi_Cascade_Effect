import pytest

from eval.parse_tree import (
    LeafNode,
    SplitNode,
    TreeParseError,
    hand_check_paths,
    iter_leaves,
    node_from_dict,
    parse_tree_text,
    path_to_leaf,
    tree_depth,
    tree_to_summary,
)

TINY_TREE = """\
|--- feat_a <= 5.00
|   |--- feat_b <= 1.50
|   |   |--- class: BENIGN
|   |--- feat_b >  1.50
|   |   |--- class: DoS Hulk
|--- feat_a >  5.00
|   |--- class: PortScan
"""


def test_parses_tiny_tree_structure():
    root = parse_tree_text(TINY_TREE)
    assert isinstance(root, SplitNode)
    assert root.feature == "feat_a"
    assert root.threshold == 5.0
    assert isinstance(root.left, SplitNode)
    assert isinstance(root.right, LeafNode)
    assert root.right.predicted_class == "PortScan"


def test_round_trip_against_hand_read_paths():
    """Round-trip: parse, then confirm 3 hand-read root-to-leaf paths
    (read directly from the fixture text above) match what the parser
    produces -- the same verification gate used against the real tree.txt."""
    root = parse_tree_text(TINY_TREE)
    leaves = {l.predicted_class: l for l in iter_leaves(root)}

    benign_path = path_to_leaf(root, leaves["BENIGN"].id)
    assert benign_path == [("feat_a", "<=", 5.0), ("feat_b", "<=", 1.5)]

    hulk_path = path_to_leaf(root, leaves["DoS Hulk"].id)
    assert hulk_path == [("feat_a", "<=", 5.0), ("feat_b", ">", 1.5)]

    portscan_path = path_to_leaf(root, leaves["PortScan"].id)
    assert portscan_path == [("feat_a", ">", 5.0)]


def test_json_round_trip_preserves_structure():
    root = parse_tree_text(TINY_TREE)
    summary = tree_to_summary(root, source_path="tree.txt")
    rebuilt = node_from_dict(summary["root"])
    assert isinstance(rebuilt, SplitNode)
    orig_leaves = sorted((l.id, l.predicted_class) for l in iter_leaves(root))
    new_leaves = sorted((l.id, l.predicted_class) for l in iter_leaves(rebuilt))
    assert orig_leaves == new_leaves


def test_summary_counts():
    root = parse_tree_text(TINY_TREE)
    summary = tree_to_summary(root, source_path="tree.txt")
    assert summary["n_leaves"] == 3
    assert summary["n_non_benign_leaves"] == 2
    assert summary["depth"] == 2
    assert set(summary["features"]) == {"feat_a", "feat_b"}
    assert summary["has_sample_counts"] is False


def test_depth_matches_longest_path():
    root = parse_tree_text(TINY_TREE)
    assert tree_depth(root) == 2


def test_hand_check_paths_runs_without_error(capsys):
    root = parse_tree_text(TINY_TREE)
    leaf_ids = [l.id for l in iter_leaves(root)]
    hand_check_paths(root, leaf_ids)
    out = capsys.readouterr().out
    assert "predicted_class" in out


def test_wrong_indent_boundary_raises():
    bad = "|-- feat_a <= 5.00\n|   |--- class: BENIGN\n"
    # 3-char marker ("|--") instead of 4 -- not a valid indent boundary
    with pytest.raises(TreeParseError):
        parse_tree_text(bad)


def test_mismatched_sibling_threshold_raises():
    bad = (
        "|--- feat_a <= 5.00\n"
        "|   |--- class: BENIGN\n"
        "|--- feat_a >  6.00\n"  # should be > 5.00 to match the left branch
        "|   |--- class: PortScan\n"
    )
    with pytest.raises(TreeParseError):
        parse_tree_text(bad)


def test_right_branch_listed_before_left_raises():
    bad = "|--- feat_a >  5.00\n|   |--- class: BENIGN\n"
    with pytest.raises(TreeParseError):
        parse_tree_text(bad)


def test_leaf_only_root_raises():
    with pytest.raises(TreeParseError):
        parse_tree_text("|--- class: BENIGN\n")
