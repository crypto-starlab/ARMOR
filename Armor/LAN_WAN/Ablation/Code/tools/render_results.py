#!/usr/bin/env python3
"""Render the w/o Dist aggregate CSV as a readable PNG and PDF table."""
import argparse
import csv
import math
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

LABELS={'wo_Dist':'w/o Dist','wo_Lazy':'w/o Lazy','wo_Coeff':'w/o Coeff','wo_Ext':'w/o Extraction','ARMOR':'ARMOR'}
FIGURE = Path(__file__).resolve().parents[2]
COLUMNS = [
    ('Parameter', 'Parameter', 165),
    ('input_blocks', 'Input blocks', 180),
    ('Enc_ms', 'Encryption', 205),
    ('HE_computation_ms', 'HE computation', 235),
    ('Dec_ms', 'Decryption', 205),
    ('Client_upload_ms', 'Client upload', 235),
    ('Server_communication_ms', 'Server communication', 325),
    ('Total_ms', 'Total', 220),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=list(LABELS), required=True)
    parser.add_argument('--csv', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    args.csv=args.csv or FIGURE/args.method/'results/results.csv'
    args.output=args.output or FIGURE/args.method/'results/results.png'

    with args.csv.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    if not rows or any(row['method'] != args.method for row in rows):
        raise ValueError('Expected nonempty result rows for the selected method')
    for row in rows:
        for key, _, _ in COLUMNS[2:]:
            if not math.isfinite(float(row[key])):
                raise ValueError(f'Nonfinite timing: {key}')
    font_dir = Path('/usr/share/fonts/truetype/dejavu')
    font = ImageFont.truetype(str(font_dir/'DejaVuSans.ttf'), 25)
    bold = ImageFont.truetype(str(font_dir/'DejaVuSans-Bold.ttf'), 25)
    title = ImageFont.truetype(str(font_dir/'DejaVuSans-Bold.ttf'), 39)
    small = ImageFont.truetype(str(font_dir/'DejaVuSans.ttf'), 23)
    margin = 45
    width = 2*margin+sum(item[2] for item in COLUMNS)
    height = 245+len(rows)*75
    image = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(image)
    draw.text((margin, 28), LABELS[args.method]+' — Norm Computation', font=title, fill='#17324d')
    draw.text((margin, 88), 'Aggregate timings | All time values are in milliseconds (ms)', font=small, fill='#46586c')
    header_y = 145
    draw.rectangle((margin, header_y, width-margin, header_y+62), fill='#17324d')
    x = margin
    for _, label, col_width in COLUMNS:
        draw.text((x+13, header_y+17), label, font=bold, fill='white')
        x += col_width
    for i, row in enumerate(rows):
        top = header_y+62+i*75
        if i % 2 == 0:
            draw.rectangle((margin, top, width-margin, top+75), fill='#edf3f9')
        x = margin
        for j, (key, _, col_width) in enumerate(COLUMNS):
            label = row[key] if j < 2 else f'{float(row[key]):,.3f}'
            left = x+13 if j < 2 else x+col_width-17-draw.textlength(label, font=font)
            draw.text((left, top+23), label, font=font, fill='#172c40')
            x += col_width
        draw.line((margin, top+75, width-margin, top+75), fill='#ccd6e0')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output, dpi=(200,200))
    pdf = args.output.with_suffix('.pdf')
    image.save(pdf, 'PDF', resolution=200)
    print(f'PNG: {args.output}')
    print(f'PDF: {pdf}')


if __name__ == '__main__':
    main()
