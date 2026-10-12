"""Export the deliberately small SVG vocabulary used by manuscript schematics.

Requires ReportLab. The SVG remains the editable master; the PDF retains native
text, vector shapes and connector paths. Unsupported SVG constructs raise errors
instead of silently becoming raster images.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from xml.etree import ElementTree as ET

from reportlab.lib.colors import HexColor
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas


def export(source: Path, destination: Path) -> None:
    root = ET.parse(source).getroot()
    _, _, width, height = map(float, root.attrib["viewBox"].split())
    pdf = canvas.Canvas(str(destination), pagesize=(width, height), pageCompression=1)
    title = root.find("{http://www.w3.org/2000/svg}title")
    pdf.setTitle(title.text if title is not None and title.text else source.stem)
    pdf.setAuthor("TrustRAG study")

    def paint(path, attrs: dict[str, str]) -> None:
        fill, stroke = attrs.get("fill", "#000000"), attrs.get("stroke", "none")
        if fill != "none":
            pdf.setFillColor(HexColor(fill))
        if stroke != "none":
            pdf.setStrokeColor(HexColor(stroke))
            pdf.setLineWidth(float(attrs.get("stroke-width", "1")))
        pdf.setLineJoin(1 if attrs.get("stroke-linejoin") == "round" else 0)
        pdf.setLineCap(1 if attrs.get("stroke-linecap") == "round" else 0)
        pdf.drawPath(path, fill=int(fill != "none"), stroke=int(stroke != "none"))

    def walk(element, inherited: dict[str, str]) -> None:
        attrs = inherited | element.attrib
        tag = element.tag.rsplit("}", 1)[-1]
        if tag in {"title", "desc"}:
            return
        if tag in {"svg", "g"}:
            for child in element:
                walk(child, attrs)
            return
        pdf.saveState()
        if tag == "rect":
            x, y = float(attrs["x"]), float(attrs["y"])
            w, h = float(attrs["width"]), float(attrs["height"])
            radius = float(attrs.get("rx", "0"))
            path = pdf.beginPath()
            if radius:
                path.roundRect(x, height - y - h, w, h, radius)
            else:
                path.rect(x, height - y - h, w, h)
            paint(path, attrs)
        elif tag == "circle":
            x, y = float(attrs["cx"]), float(attrs["cy"])
            radius = float(attrs["r"])
            path = pdf.beginPath()
            path.ellipse(x - radius, height - y - radius, 2 * radius, 2 * radius)
            paint(path, attrs)
        elif tag == "ellipse":
            x, y = float(attrs["cx"]), float(attrs["cy"])
            rx, ry = float(attrs["rx"]), float(attrs["ry"])
            path = pdf.beginPath()
            path.ellipse(x - rx, height - y - ry, 2 * rx, 2 * ry)
            paint(path, attrs)
        elif tag == "polygon":
            numbers = list(map(float, re.findall(r"-?\d+(?:\.\d+)?", attrs["points"])))
            points = list(zip(numbers[::2], numbers[1::2]))
            path = pdf.beginPath()
            path.moveTo(points[0][0], height - points[0][1])
            for x, y in points[1:]:
                path.lineTo(x, height - y)
            path.close()
            paint(path, attrs)
        elif tag == "path":
            tokens = re.findall(r"[MmLlHhVvCcZz]|-?\d+(?:\.\d+)?", attrs["d"])
            path = pdf.beginPath()
            index, x, y = 0, 0.0, 0.0
            while index < len(tokens):
                command = tokens[index]
                index += 1
                if command in {"M", "L"}:
                    x, y = float(tokens[index]), float(tokens[index + 1])
                    index += 2
                    if command == "M":
                        path.moveTo(x, height - y)
                    else:
                        path.lineTo(x, height - y)
                elif command == "H":
                    x = float(tokens[index])
                    index += 1
                    path.lineTo(x, height - y)
                elif command == "V":
                    y = float(tokens[index])
                    index += 1
                    path.lineTo(x, height - y)
                elif command == "C":
                    x1, y1, x2, y2, x, y = map(float, tokens[index : index + 6])
                    index += 6
                    path.curveTo(x1, height - y1, x2, height - y2, x, height - y)
                elif command in {"Z", "z"}:
                    path.close()
                else:
                    raise ValueError(f"Unsupported path command: {command}")
            paint(path, attrs)
        elif tag == "text":
            if len(element):
                raise ValueError("Use individual text elements, not nested text spans")
            label = element.text or ""
            size = float(attrs.get("font-size", "6.2"))
            weight = attrs.get("font-weight", "400")
            font = "Helvetica-Bold" if weight in {"bold", "700"} else "Helvetica"
            if attrs.get("font-family", "Helvetica") != "Helvetica":
                raise ValueError("This exporter is configured for Helvetica")
            pdf.setFont(font, size)
            pdf.setFillColor(HexColor(attrs.get("fill", "#253645")))
            x, y = float(attrs["x"]), height - float(attrs["y"])
            anchor = attrs.get("text-anchor", "start")
            if anchor == "middle":
                x -= stringWidth(label, font, size) / 2
            elif anchor == "end":
                x -= stringWidth(label, font, size)
            pdf.drawString(x, y, label)
        else:
            raise ValueError(f"Unsupported SVG element: {tag}")
        pdf.restoreState()

    walk(root, {})
    pdf.showPage()
    pdf.save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("svg", type=Path)
    parser.add_argument("--pdf", type=Path)
    args = parser.parse_args()
    export(args.svg, args.pdf or args.svg.with_suffix(".pdf"))
