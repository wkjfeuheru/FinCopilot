import base64
import io
import json
import zipfile

from finharness.compute.worker import _absolutize_images, _docx_export_handler

_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\xcf\xc0"
    b"\x00\x00\x03\x01\x01\x00\x18\xdd\x8d\xb0\x00\x00\x00\x00IEND\xaeB`\x82"
)


def test_docx_worker_handler_returns_a_named_blob(tmp_path):
    (tmp_path / "request.json").write_text(
        json.dumps({"markdown": "# 测试报告\n\n正文", "topic": "测试"}), encoding="utf-8"
    )

    result = _docx_export_handler({"kind": "docx_export"}, tmp_path)

    assert base64.b64decode(result["blobs"]["report.docx"]).startswith(b"PK")


def test_docx_worker_embeds_packaged_images(tmp_path):
    """随包送达的图表要被真正嵌入，而不是降级成"[图片缺失]"。"""
    (tmp_path / "img_abc.png").write_bytes(_PNG)
    (tmp_path / "request.json").write_text(
        json.dumps({"markdown": "![图](<img_abc.png>)", "topic": "带图"}), encoding="utf-8"
    )

    result = _docx_export_handler({"kind": "docx_export"}, tmp_path)

    docx = base64.b64decode(result["blobs"]["report.docx"])
    with zipfile.ZipFile(io.BytesIO(docx)) as bundle:
        media = [name for name in bundle.namelist() if name.startswith("word/media/")]
        assert media, "图片必须进入 docx 的 word/media/"
        assert bundle.read(media[0]) == _PNG


def test_absolutize_images_resolves_packaged_names_and_leaves_missing_ones(tmp_path):
    (tmp_path / "img_abc.png").write_bytes(b"x")
    markdown = "![A](<img_abc.png>) ![B](absent.png)"

    resolved = _absolutize_images(markdown, tmp_path)

    assert str(tmp_path / "img_abc.png") in resolved
    assert "![B](absent.png)" in resolved
