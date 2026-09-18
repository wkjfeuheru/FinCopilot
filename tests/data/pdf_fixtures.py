"""供 data adapter 使用的共享测试辅助。

保持为普通模块（而非 conftest fixture），这样 adapter 自身的测试和编排测试
都能构建同一个 PDF，而无需一方导入另一方的测试模块。
"""

from __future__ import annotations


def make_pdf(text: str = "Hello Direct PDF") -> bytes:
    """一个有效的单页 PDF，内含真实的文本对象。

    采用构造方式而非以二进制文件检入，这样抽取出的字符串在对它做断言的
    测试中直接可见。
    """
    content = ("BT /F1 24 Tf 72 700 Td (%s) Tj ET" % text).encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length %d >>\nstream\n%s\nendstream" % (len(content), content),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for index, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % index + body + b"\nendobj\n"
    xref_at = len(out)
    out += b"xref\n0 %d\n" % (len(objects) + 1)
    out += b"0000000000 65535 f \n"
    for offset in offsets:
        out += b"%010d 00000 n \n" % offset
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (
        len(objects) + 1,
        xref_at,
    )
    return bytes(out)


def make_multi_page_pdf(pages: list[str]) -> bytes:
    """一个有效的多页 PDF，每页一个文本对象。

    分页读取（``read_pdf``）依赖真实的页边界，因此这里按页码生成 Pages 树，而不是
    复用单页构造再拼接——页码属于 PDF 结构，不属于文本内容。
    """
    page_count = len(pages)
    # 对象编号：1=Catalog，2=Pages，3..(2+n)=Page，随后是每页的 Contents 与共用的 Font。
    first_page_obj = 3
    contents_start = first_page_obj + page_count
    font_obj = contents_start + page_count

    kids = " ".join(f"{first_page_obj + i} 0 R" for i in range(page_count))
    objects: list[bytes] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [%s] /Count %d >>" % (kids.encode(), page_count),
    ]
    for index in range(page_count):
        objects.append(
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Contents %d 0 R /Resources << /Font << /F1 %d 0 R >> >> >>"
            % (contents_start + index, font_obj)
        )
    for text in pages:
        content = ("BT /F1 24 Tf 72 700 Td (%s) Tj ET" % text).encode()
        objects.append(
            b"<< /Length %d >>\nstream\n%s\nendstream" % (len(content), content)
        )
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for index, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % index + body + b"\nendobj\n"
    xref_at = len(out)
    out += b"xref\n0 %d\n" % (len(objects) + 1)
    out += b"0000000000 65535 f \n"
    for offset in offsets:
        out += b"%010d 00000 n \n" % offset
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (
        len(objects) + 1,
        xref_at,
    )
    return bytes(out)
