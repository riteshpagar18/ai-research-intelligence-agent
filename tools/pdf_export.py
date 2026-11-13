"""Create a simple, readable PDF version of a research report."""

from io import BytesIO
import re
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
)


def _inline_markup(text: str) -> str:
    safe = escape(text)
    safe = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", safe)
    safe = re.sub(r"\*(.+?)\*", r"<i>\1</i>", safe)
    return safe


def create_report_pdf(report: str, sources: list[dict]) -> bytes:
    """Render report text and its linked source list into PDF bytes."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=0.75 * inch,
        leftMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.65 * inch,
        title="AI Research Report",
        author="AI Research & Intelligence Agent",
    )
    base = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ReportTitle", parent=base["Title"], alignment=TA_CENTER, textColor=colors.HexColor("#17324D")
    )
    body_style = ParagraphStyle(
        "ReportBody", parent=base["BodyText"], leading=14, spaceAfter=7
    )
    story = [Paragraph("AI Research Report", title_style), Spacer(1, 12)]
    bullet_items = []

    def flush_bullets():
        if bullet_items:
            story.append(ListFlowable(
                [ListItem(Paragraph(item, body_style)) for item in bullet_items],
                bulletType="bullet", leftIndent=18, bulletFontName="Helvetica",
            ))
            story.append(Spacer(1, 6))
            bullet_items.clear()

    for raw_line in report.splitlines():
        line = raw_line.strip()
        if not line:
            flush_bullets()
            continue
        heading = re.match(r"^(#{1,3})\s+(.*)$", line)
        if heading:
            flush_bullets()
            level = len(heading.group(1))
            story.append(Paragraph(_inline_markup(heading.group(2)), base[f"Heading{level}"]))
        elif re.match(r"^[-*]\s+", line):
            bullet_items.append(_inline_markup(re.sub(r"^[-*]\s+", "", line)))
        else:
            flush_bullets()
            story.append(Paragraph(_inline_markup(line), body_style))
    flush_bullets()

    if sources:
        story.extend([PageBreak(), Paragraph("Sources", base["Heading1"])])
        for source in sources:
            source_id = escape(source.get("source_id", ""))
            title = escape(source.get("title", "Untitled"))
            raw_url = source.get("url", "")
            safe_url = escape(raw_url, {'"': "&quot;"})
            title_markup = (
                f'<link href="{safe_url}" color="#1769aa">{title}</link>'
                if raw_url.startswith(("https://", "http://"))
                else title
            )
            story.append(Paragraph(
                f"<b>[{source_id}] {title_markup}</b>",
                body_style,
            ))

    def add_page_number(canvas, document):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.grey)
        canvas.drawRightString(letter[0] - 0.75 * inch, 0.35 * inch, str(document.page))
        canvas.restoreState()

    doc.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)
    return buffer.getvalue()
