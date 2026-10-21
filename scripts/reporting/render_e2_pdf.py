from pathlib import Path
from io import BytesIO
from html import escape
from docx import Document
from docx.text.paragraph import Paragraph as WordParagraph
from docx.table import Table as WordTable
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from pypdf import PdfReader
from pdf2image import convert_from_path

import argparse
parser=argparse.ArgumentParser()
parser.add_argument('--root',type=Path,required=True)
args=parser.parse_args()
ROOT=args.root
pdfmetrics.registerFont(TTFont('Calibri','C:/Windows/Fonts/calibri.ttf'))
pdfmetrics.registerFont(TTFont('CalibriBold','C:/Windows/Fonts/calibrib.ttf'))
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='BodyE2',fontName='Calibri',fontSize=10.2,leading=13.2,spaceAfter=7))
styles.add(ParagraphStyle(name='TitleE2',fontName='CalibriBold',fontSize=25,leading=29,spaceAfter=12))
styles.add(ParagraphStyle(name='SubE2',fontName='Calibri',fontSize=13,leading=17,spaceAfter=13))
styles.add(ParagraphStyle(name='H1E2',fontName='CalibriBold',fontSize=17,leading=21,spaceBefore=4,spaceAfter=10,keepWithNext=True))
styles.add(ParagraphStyle(name='H2E2',fontName='CalibriBold',fontSize=12,leading=15,spaceBefore=7,spaceAfter=6,keepWithNext=True))
styles.add(ParagraphStyle(name='CaptionE2',fontName='Calibri',fontSize=8.8,leading=11,spaceAfter=8))
styles.add(ParagraphStyle(name='CellE2',fontName='Calibri',fontSize=8.5,leading=10.5))
styles.add(ParagraphStyle(name='CellHeadE2',fontName='CalibriBold',fontSize=8.5,leading=10.5))
for style in styles.byName.values():
    style.textColor=colors.black
document=Document(ROOT/'outputs/E2Evaluation.docx')
story=[]
width=A4[0]-92
for element in document.element.body:
    tag=element.tag.split('}')[-1]
    if tag=='p':
        para=WordParagraph(element,document)
        if element.xpath('.//w:br[@w:type="page"]'):
            story.append(PageBreak()); continue
        blips=element.xpath('.//a:blip')
        if blips:
            rid=blips[0].get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed')
            blob=document.part.related_parts[rid].blob
            picture=Image(BytesIO(blob))
            ratio=picture.imageHeight/picture.imageWidth
            picture.drawWidth=width; picture.drawHeight=width*ratio
            story.append(picture)
        elif para.text.strip():
            name=para.style.name
            style={'Title':'TitleE2','Subtitle':'SubE2','Heading 1':'H1E2','Heading 2':'H2E2','Caption':'CaptionE2'}.get(name,'BodyE2')
            story.append(Paragraph(escape(para.text),styles[style]))
    elif tag=='tbl':
        table=WordTable(element,document)
        ratios=[cell.width or 1 for cell in table.rows[0].cells]
        widths=[width*value/sum(ratios) for value in ratios]
        rows=[[Paragraph(escape(cell.text),styles['CellHeadE2' if i==0 else 'CellE2']) for cell in row.cells] for i,row in enumerate(table.rows)]
        result=Table(rows,colWidths=widths,repeatRows=1,hAlign='LEFT')
        result.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#E7E6E6')),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5),('LINEBELOW',(0,0),(-1,0),.5,colors.grey),('LINEBELOW',(0,1),(-1,-1),.25,colors.HexColor('#DDDDDD'))]))
        story.extend([result,Spacer(1,9)])
def furniture(canvas,doc):
    canvas.saveState()
    canvas.setFillColor(colors.black)
    canvas.setFont('Calibri',8)
    canvas.drawString(46,A4[1]-25,'ICU MULTIMODAL BENCHMARK     |     E2 EVALUATION')
    canvas.drawString(46,24,'21 October 2026     •     Research evaluation')
    canvas.drawRightString(A4[0]-46,24,str(doc.page))
    canvas.restoreState()
dest=ROOT/'outputs/E2Evaluation.pdf'
SimpleDocTemplate(str(dest),pagesize=A4,leftMargin=46,rightMargin=46,topMargin=45,bottomMargin=43,title='Evaluation of E2 Unimodal Encoders',author='ICU multimodal benchmark').build(story,onFirstPage=furniture,onLaterPages=furniture)
pdf=PdfReader(dest)
folder=ROOT/'work/e2render'; folder.mkdir(exist_ok=True)
for i,page in enumerate(pdf.pages):
    print('PAGE',i+1,'WORDS',len(page.extract_text().split()))
print('PDF_CREATED',len(pdf.pages),'pages')
