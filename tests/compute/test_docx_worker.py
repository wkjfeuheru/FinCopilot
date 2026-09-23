import base64
import json

from finharness.compute.worker import _docx_export_handler


def test_docx_worker_handler_returns_a_named_blob(tmp_path):
    (tmp_path / "request.json").write_text(
        json.dumps({"markdown": "# 测试报告\n\n正文", "topic": "测试"}), encoding="utf-8"
    )

    result = _docx_export_handler({"kind": "docx_export"}, tmp_path)

    assert base64.b64decode(result["blobs"]["report.docx"]).startswith(b"PK")
