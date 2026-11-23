"""增量归并缺键处理测试。

覆盖剧情：同一批增量里 K 和 _ 是来源中的有效归并键，原始记录 A、B 均缺键。
缺键行必须逐条保留明细（金额、原始标识、处理状态），不能被合到同一条记录，
也不能把下划线 ``_`` 当成缺键占位键；补键续补后金额分别落到不同行，
已提交的正常行不会因重投而二次累计。
"""
from __future__ import annotations

import pytest

from app.fusion_engine import FusionEngine


SOURCE = "SRC-DELTA"


def _find_row(records: list[dict], join_value: object) -> dict | None:
    for record in records:
        if record.get("_source") == SOURCE and record.get("id") == join_value:
            return record
    return None


@pytest.fixture
def engine_with_job() -> tuple[FusionEngine, int]:
    """构造一个带存量数据的融合作业：合法键 K 与合法键 _ 已各占一行。"""
    engine = FusionEngine()
    engine.register_source(
        source_code=SOURCE, source_name="delta_source", source_type="MySQL",
        connection_string="conn://delta",
    )
    job = engine.create_fusion_job(
        target_wide_table="WT_DELTA_KEY",
        participant_sources=[SOURCE],
        join_key="id",
    )
    job.merged_records = [
        {"_source": SOURCE, "id": "K", "amount": 100},
        {"_source": SOURCE, "id": "_", "amount": 200},
    ]
    return engine, job.job_id


class TestMakeMergeKey:
    """归并键构造：缺键与合法下划线键必须可区分。"""

    def test_missing_key_returns_none(self) -> None:
        assert FusionEngine._make_merge_key({"_source": SOURCE, "amount": 1}, "id") is None
        assert FusionEngine._make_merge_key({"_source": SOURCE, "id": None}, "id") is None
        assert FusionEngine._make_merge_key({"_source": SOURCE, "id": ""}, "id") is None

    def test_underscore_is_a_valid_key(self) -> None:
        key = FusionEngine._make_merge_key({"_source": SOURCE, "id": "_"}, "id")
        assert key == (SOURCE, "_")
        assert FusionEngine._render_merge_key(key) == f"{SOURCE}#_"

    def test_normal_key_tuple(self) -> None:
        assert FusionEngine._make_merge_key({"_source": SOURCE, "id": "K"}, "id") == (SOURCE, "K")


class TestKeylessDeltaMerge:
    """缺键增量行的归并入口处理。"""

    def test_keyless_rows_keep_separate_details_and_fail_explicitly(
        self, engine_with_job: tuple[FusionEngine, int]
    ) -> None:
        engine, job_id = engine_with_job
        wm = engine.create_watermark(data_source=SOURCE, current_watermark="2026-11-23T00:00:00Z")
        before_wm = wm.current_watermark

        deltas = [
            {"_raw_id": "A", "_source": SOURCE, "amount": 20, "_op": "upsert"},
            {"_raw_id": "B", "_source": SOURCE, "amount": 30, "_op": "upsert"},
            {"_raw_id": "N1", "_source": SOURCE, "id": "N", "amount": 7, "_op": "upsert"},
        ]
        result = engine.execute_delta_merge(job_id, deltas, watermark_record_id=wm.record_id)

        # 整批存在明确失败：ok 为 False，失败位置精确到批次下标和原始标识
        assert result["ok"] is False
        assert {(f["index"], f["raw_id"]) for f in result["failures"]} == {(0, "A"), (1, "B")}

        # 缺键行逐条可查：金额、原始标识、处理状态完整，键值是 None 而不是下划线
        detail_a = engine.get_keyless_detail(job_id, "A")
        detail_b = engine.get_keyless_detail(job_id, "B")
        assert detail_a is not None and detail_b is not None
        assert detail_a["amount"] == 20 and detail_a["status"] == "待核键"
        assert detail_a["join_key_value"] is None
        assert detail_b["amount"] == 30 and detail_b["status"] == "待核键"
        assert detail_b["join_key_value"] is None
        pending = engine.list_keyless_details(job_id, status="待核键")
        assert {d["raw_id"] for d in pending} == {"A", "B"}

        # 已生效快照范围：只有正常行 N 进入，A、B 留在待核区，合法 K/_ 行未被覆盖
        scope = result["snapshot_scope"]
        assert scope["effective_record_count"] == 3
        assert set(scope["effective_raw_ids"]) == {"N1"}
        assert set(scope["pending_raw_ids"]) == {"A", "B"}
        job = engine._fusion_jobs[job_id]
        assert _find_row(job.merged_records, "K")["amount"] == 100
        assert _find_row(job.merged_records, "_")["amount"] == 200
        assert _find_row(job.merged_records, "N")["amount"] == 7
        assert _find_row(job.merged_records, "A") is None
        assert _find_row(job.merged_records, "B") is None
        # 两条缺键行没有被合到同一条记录
        assert result["total_after_merge"] == 3

        # 存在失败时水位不推进，正常行与待核行的提交边界不混淆
        assert wm.current_watermark == before_wm

        # 批次结果关联缺键行明细，补交请求能明确哪些原始值可续补
        stored = engine.get_merge_batch(result["batch_code"])
        assert stored is not None
        assert {d["raw_id"] for d in stored["keyless_details"]} == {"A", "B"}

    def test_keyless_delete_is_rejected_but_detail_kept(
        self, engine_with_job: tuple[FusionEngine, int]
    ) -> None:
        engine, job_id = engine_with_job
        result = engine.execute_delta_merge(job_id, [
            {"_raw_id": "D1", "_source": SOURCE, "amount": 9, "_op": "delete"},
        ])
        assert result["ok"] is False
        assert "delete" in result["failures"][0]["reason"]
        detail = engine.get_keyless_detail(job_id, "D1")
        assert detail is not None and detail["amount"] == 9

    def test_repair_with_distinct_keys_lands_amounts_on_different_rows(
        self, engine_with_job: tuple[FusionEngine, int]
    ) -> None:
        engine, job_id = engine_with_job
        first = engine.execute_delta_merge(job_id, [
            {"_raw_id": "A", "_source": SOURCE, "amount": 20, "_op": "upsert"},
            {"_raw_id": "B", "_source": SOURCE, "amount": 30, "_op": "upsert"},
        ])
        assert first["ok"] is False

        # 分别补不同的键后重试
        retry = engine.execute_delta_merge(job_id, [
            {"_raw_id": "A", "_source": SOURCE, "id": "KA", "amount": 20, "_op": "upsert"},
            {"_raw_id": "B", "_source": SOURCE, "id": "KB", "amount": 30, "_op": "upsert"},
        ])
        assert retry["ok"] is True
        assert retry["inserted"] == 2
        assert retry["failures"] == []

        job = engine._fusion_jobs[job_id]
        row_a = _find_row(job.merged_records, "KA")
        row_b = _find_row(job.merged_records, "KB")
        assert row_a is not None and row_b is not None
        assert row_a is not row_b
        assert row_a["amount"] == 20 and row_b["amount"] == 30

        # 合法下划线键行自始至终未被 A、B 任一条覆盖
        assert _find_row(job.merged_records, "_")["amount"] == 200
        assert _find_row(job.merged_records, "K")["amount"] == 100

        # 明细状态翻成已补键，并记录最终归并键
        detail_a = engine.get_keyless_detail(job_id, "A")
        detail_b = engine.get_keyless_detail(job_id, "B")
        assert detail_a["status"] == "已补键" and detail_a["resolved_merge_key"] == f"{SOURCE}#KA"
        assert detail_b["status"] == "已补键" and detail_b["resolved_merge_key"] == f"{SOURCE}#KB"
        assert engine.list_keyless_details(job_id, status="待核键") == []

    def test_committed_rows_are_not_accumulated_twice(
        self, engine_with_job: tuple[FusionEngine, int]
    ) -> None:
        engine, job_id = engine_with_job
        engine.execute_delta_merge(job_id, [
            {"_raw_id": "N1", "_source": SOURCE, "id": "N", "amount": 7, "_op": "upsert"},
            {"_raw_id": "A", "_source": SOURCE, "amount": 20, "_op": "upsert"},
        ])
        # A 补键生效
        engine.execute_delta_merge(job_id, [
            {"_raw_id": "A", "_source": SOURCE, "id": "KA", "amount": 20, "_op": "upsert"},
        ])
        job = engine._fusion_jobs[job_id]
        total_after_repair = len(job.merged_records)

        # 正常行 N1 与补键行 A 再次重投：幂等跳过，不二次累计
        again = engine.execute_delta_merge(job_id, [
            {"_raw_id": "N1", "_source": SOURCE, "id": "N", "amount": 7, "_op": "upsert"},
            {"_raw_id": "A", "_source": SOURCE, "id": "KA", "amount": 20, "_op": "upsert"},
        ])
        assert again["duplicates"] == 2
        assert again["inserted"] == 0 and again["updated"] == 0
        assert len(job.merged_records) == total_after_repair
        assert _find_row(job.merged_records, "KA")["amount"] == 20

    def test_underscore_key_row_is_never_treated_as_keyless(
        self, engine_with_job: tuple[FusionEngine, int]
    ) -> None:
        engine, job_id = engine_with_job
        result = engine.execute_delta_merge(job_id, [
            {"_raw_id": "U1", "_source": SOURCE, "id": "_", "amount": 201, "_op": "upsert"},
        ])
        assert result["ok"] is True
        assert result["keyless_count"] == 0
        # 合法 _ 行走正常 upsert，更新的是 _ 自己那一行，与缺键明细无关
        assert _find_row(engine._fusion_jobs[job_id].merged_records, "_")["amount"] == 201
        assert engine.list_keyless_details(job_id) == []


class TestRepairKeylessRecord:
    """补键续补专用入口。"""

    def test_repair_inserts_then_rejects_repeat_and_empty(
        self, engine_with_job: tuple[FusionEngine, int]
    ) -> None:
        engine, job_id = engine_with_job
        engine.execute_delta_merge(job_id, [
            {"_raw_id": "C", "_source": SOURCE, "amount": 40, "_op": "upsert"},
        ])
        repaired = engine.repair_keyless_record(job_id, "C", "KC")
        assert repaired["ok"] is True and repaired["action"] == "inserted"
        assert repaired["merge_key"] == f"{SOURCE}#KC"
        assert _find_row(engine._fusion_jobs[job_id].merged_records, "KC")["amount"] == 40

        # 已补键行重复处理被拒绝，避免二次累计
        repeat = engine.repair_keyless_record(job_id, "C", "KC2")
        assert repeat["ok"] is False
        # 空键值仍然算缺键，不能写成占位下划线键
        empty = engine.repair_keyless_record(job_id, "C", "")
        assert empty["ok"] is False

    def test_repair_unknown_detail_returns_error(
        self, engine_with_job: tuple[FusionEngine, int]
    ) -> None:
        engine, job_id = engine_with_job
        assert engine.repair_keyless_record(job_id, "GHOST", "KX")["ok"] is False

    def test_repair_to_underscore_key_is_legal(
        self, engine_with_job: tuple[FusionEngine, int]
    ) -> None:
        engine, job_id = engine_with_job
        engine.execute_delta_merge(job_id, [
            {"_raw_id": "E", "_source": SOURCE, "amount": 50, "_op": "upsert"},
        ])
        # 下划线是合法键值：补成 _ 时走正常键通道，而不是被当成仍然缺键
        repaired = engine.repair_keyless_record(job_id, "E", "_")
        assert repaired["ok"] is True
        assert repaired["merge_key"] == f"{SOURCE}#_"
        detail = engine.get_keyless_detail(job_id, "E")
        assert detail["status"] == "已补键" and detail["join_key_value"] == "_"
