from controlplane.record import EscalationRecord, PacketWindowSummary

from eval.run_agent_pipeline import stratified_sample


def _rec(flow_id):
    return EscalationRecord(
        flow_id=flow_id,
        trigger_reasons=[],
        features={"x": 1.0},
        packet_window=PacketWindowSummary(packet_count=1, byte_count=1, duration_us=1),
        selector_config_hash="h",
    )


def test_stratified_sample_takes_per_class_and_sorts_by_label():
    records = [_rec(f"f{i}") for i in range(6)]
    manifest = {
        "f0": "Bot", "f1": "Bot", "f2": "BENIGN",
        "f3": "BENIGN", "f4": "DDoS", "f5": "DDoS",
    }
    sample = stratified_sample(records, manifest, per_class=1)
    labels = sorted(manifest[r.flow_id] for r in sample)
    assert labels == ["BENIGN", "Bot", "DDoS"]


def test_stratified_sample_caps_at_per_class_even_with_more_available():
    records = [_rec(f"f{i}") for i in range(4)]
    manifest = {f"f{i}": "Bot" for i in range(4)}
    sample = stratified_sample(records, manifest, per_class=2)
    assert len(sample) == 2


def test_stratified_sample_handles_unknown_labels_gracefully():
    records = [_rec("f0")]
    sample = stratified_sample(records, manifest={}, per_class=5)
    assert len(sample) == 1
