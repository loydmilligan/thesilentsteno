"""
PDF Generation System

Professional PDF document generation using ReportLab with customizable templates and styles.
"""

import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import json

# ReportLab imports
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import Color, black, blue, gray
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.platypus.tableofcontents import TableOfContents
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY

logger = logging.getLogger(__name__)

@dataclass
class DocumentStyle:
    """PDF document styling configuration"""
    font_family: str = "Helvetica"
    font_size: int = 12
    title_font_size: int = 18
    header_font_size: int = 14
    line_height: float = 1.2
    margin_left: float = 1.0
    margin_right: float = 1.0
    margin_top: float = 1.0
    margin_bottom: float = 1.0
    header_color: Color = blue
    text_color: Color = black
    table_border_color: Color = gray
    page_size: Tuple[float, float] = letter

@dataclass
class PDFConfig:
    """PDF generation configuration"""
    include_cover_page: bool = True
    include_table_of_contents: bool = True
    include_transcript: bool = True
    include_analysis: bool = True
    include_participant_stats: bool = True
    include_appendix: bool = True
    page_numbers: bool = True
    watermark: Optional[str] = None
    logo_path: Optional[str] = None
    max_pages: int = 100
    compress: bool = True

@dataclass
class PDFTemplate:
    """PDF template configuration"""
    name: str
    description: str
    style: DocumentStyle
    config: PDFConfig
    custom_sections: List[str] = field(default_factory=list)

class PDFGenerator:
    """Professional PDF generation system"""
    
    def __init__(self, config: PDFConfig, style: Optional[DocumentStyle] = None):
        self.config = config
        self.style = style or DocumentStyle()
        self.styles = getSampleStyleSheet()
        self._customize_styles()
        
    def _customize_styles(self):
        """Customize ReportLab styles"""
        # Title style
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Title'],
            fontSize=self.style.title_font_size,
            textColor=self.style.header_color,
            fontName=self.style.font_family + '-Bold',
            alignment=TA_CENTER,
            spaceAfter=30
        ))
        
        # Header style
        self.styles.add(ParagraphStyle(
            name='CustomHeader',
            parent=self.styles['Heading1'],
            fontSize=self.style.header_font_size,
            textColor=self.style.header_color,
            fontName=self.style.font_family + '-Bold',
            spaceBefore=20,
            spaceAfter=12
        ))
        
        # Body style
        self.styles.add(ParagraphStyle(
            name='CustomBody',
            parent=self.styles['Normal'],
            fontSize=self.style.font_size,
            textColor=self.style.text_color,
            fontName=self.style.font_family,
            alignment=TA_JUSTIFY,
            leading=self.style.font_size * self.style.line_height
        ))
        
        # Speaker style
        self.styles.add(ParagraphStyle(
            name='SpeakerName',
            parent=self.styles['Normal'],
            fontSize=self.style.font_size,
            textColor=self.style.header_color,
            fontName=self.style.font_family + '-Bold',
            leftIndent=0,
            spaceBefore=6
        ))
        
        # Transcript style
        self.styles.add(ParagraphStyle(
            name='TranscriptText',
            parent=self.styles['Normal'],
            fontSize=self.style.font_size - 1,
            textColor=self.style.text_color,
            fontName=self.style.font_family,
            leftIndent=20,
            rightIndent=20,
            spaceBefore=3,
            spaceAfter=3
        ))
    
    def generate_session_pdf(self, session_id: str, output_path: Optional[str] = None) -> str:
        """Generate complete session PDF"""
        try:
            # Load session data
            session_data = self._load_session_data(session_id)
            
            # Create output path
            if output_path is None:
                output_path = f"/tmp/session_{session_id}.pdf"
            
            # Create PDF document
            doc = SimpleDocTemplate(
                output_path,
                pagesize=self.style.page_size,
                rightMargin=self.style.margin_right * inch,
                leftMargin=self.style.margin_left * inch,
                topMargin=self.style.margin_top * inch,
                bottomMargin=self.style.margin_bottom * inch
            )
            
            # Build document content
            story = []
            
            if self.config.include_cover_page:
                story.extend(self._create_cover_page(session_data))
            
            if self.config.include_table_of_contents:
                story.extend(self._create_table_of_contents())
            
            if self.config.include_transcript:
                story.extend(self._create_transcript_section(session_data))
            
            if self.config.include_analysis:
                story.extend(self._create_analysis_section(session_data))
            
            if self.config.include_participant_stats:
                story.extend(self._create_participant_section(session_data))
            
            if self.config.include_appendix:
                story.extend(self._create_appendix_section(session_data))
            
            # Build PDF
            doc.build(story)
            
            logger.info(f"PDF generated successfully: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"PDF generation failed for session {session_id}: {str(e)}")
            raise
    
    def _load_session_data(self, session_id: str) -> Dict[str, Any]:
        """Load session data from database"""
        # This would integrate with the database system
        # For now, return mock data
        return {
            'session_id': session_id,
            'title': f'Meeting Session {session_id}',
            'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'duration': '45 minutes',
            'participant_count': 3,
            'participants': [
                {'name': 'John Smith', 'speaking_time': '15 min', 'percentage': 33},
                {'name': 'Jane Doe', 'speaking_time': '20 min', 'percentage': 44},
                {'name': 'Bob Johnson', 'speaking_time': '10 min', 'percentage': 23}
            ],
            'transcript': [
                {'speaker': 'John Smith', 'time': '00:00:30', 'text': 'Welcome everyone to today\'s meeting. Let\'s start with the project status update.'},
                {'speaker': 'Jane Doe', 'time': '00:01:15', 'text': 'Thanks John. The development phase is progressing well. We\'ve completed 75% of the core features.'},
                {'speaker': 'Bob Johnson', 'time': '00:02:00', 'text': 'That\'s great news. What about the testing phase? Are we still on track for the deadline?'},
                {'speaker': 'Jane Doe', 'time': '00:02:30', 'text': 'Yes, we should be able to start testing next week. The QA team is already preparing the test cases.'}
            ],
            'analysis': {
                'summary': 'Product development meeting discussing project progress, timeline, and next steps.',
                'key_topics': ['Project Status', 'Development Progress', 'Testing Timeline', 'QA Preparation'],
                'action_items': [
                    'Jane to provide detailed progress report by Friday',
                    'Bob to coordinate with QA team for test planning',
                    'John to schedule follow-up meeting for next week'
                ],
                'sentiment': 'Positive - team is confident about meeting deadlines'
            }
        }
    
    def _create_cover_page(self, session_data: Dict[str, Any]) -> List[Any]:
        """Create PDF cover page"""
        story = []
        
        # Title
        title = Paragraph(session_data['title'], self.styles['CustomTitle'])
        story.append(title)
        story.append(Spacer(1, 0.5 * inch))
        
        # Meeting details table
        details_data = [
            ['Date:', session_data['date']],
            ['Duration:', session_data['duration']],
            ['Participants:', str(session_data['participant_count'])],
            ['Session ID:', session_data['session_id']]
        ]
        
        details_table = Table(details_data, colWidths=[2*inch, 4*inch])
        details_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('FONTNAME', (0, 0), (0, -1), self.style.font_family + '-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), self.style.font_size),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ]))
        
        story.append(details_table)
        story.append(PageBreak())
        
        return story
    
    def _create_table_of_contents(self) -> List[Any]:
        """Create table of contents"""
        story = []
        
        toc_header = Paragraph("Table of Contents", self.styles['CustomTitle'])
        story.append(toc_header)
        story.append(Spacer(1, 0.3 * inch))
        
        # TOC entries
        toc_data = [
            ['1. Meeting Summary', '3'],
            ['2. Full Transcript', '4'],
            ['3. Analysis & Insights', '8'],
            ['4. Participant Statistics', '10'],
            ['5. Appendix', '12']
        ]
        
        toc_table = Table(toc_data, colWidths=[5*inch, 1*inch])
        toc_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (0, -1), 'LEFT'),
            ('ALIGN', (1, 0), (1, -1), 'RIGHT'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('FONTNAME', (0, 0), (-1, -1), self.style.font_family),
            ('FONTSIZE', (0, 0), (-1, -1), self.style.font_size),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('LINEBELOW', (0, 0), (-1, -1), 0.5, self.style.table_border_color),
        ]))
        
        story.append(toc_table)
        story.append(PageBreak())
        
        return story
    
    def _create_transcript_section(self, session_data: Dict[str, Any]) -> List[Any]:
        """Create transcript section"""
        story = []
        
        # Section header
        header = Paragraph("Full Transcript", self.styles['CustomHeader'])
        story.append(header)
        story.append(Spacer(1, 0.2 * inch))
        
        # Transcript entries
        for entry in session_data['transcript']:
            # Speaker name and time
            speaker_line = f"<b>{entry['speaker']}</b> [{entry['time']}]"
            speaker_para = Paragraph(speaker_line, self.styles['SpeakerName'])
            story.append(speaker_para)
            
            # Transcript text
            text_para = Paragraph(entry['text'], self.styles['TranscriptText'])
            story.append(text_para)
            story.append(Spacer(1, 0.1 * inch))
        
        story.append(PageBreak())
        return story
    
    def _create_analysis_section(self, session_data: Dict[str, Any]) -> List[Any]:
        """Create analysis section"""
        story = []
        
        # Section header
        header = Paragraph("Analysis & Insights", self.styles['CustomHeader'])
        story.append(header)
        story.append(Spacer(1, 0.2 * inch))
        
        analysis = session_data['analysis']
        
        # Summary
        summary_header = Paragraph("Meeting Summary", self.styles['CustomHeader'])
        story.append(summary_header)
        summary_text = Paragraph(analysis['summary'], self.styles['CustomBody'])
        story.append(summary_text)
        story.append(Spacer(1, 0.2 * inch))
        
        # Key topics
        topics_header = Paragraph("Key Topics", self.styles['CustomHeader'])
        story.append(topics_header)
        for topic in analysis['key_topics']:
            topic_text = Paragraph(f"• {topic}", self.styles['CustomBody'])
            story.append(topic_text)
        story.append(Spacer(1, 0.2 * inch))
        
        # Action items
        actions_header = Paragraph("Action Items", self.styles['CustomHeader'])
        story.append(actions_header)
        for action in analysis['action_items']:
            action_text = Paragraph(f"• {action}", self.styles['CustomBody'])
            story.append(action_text)
        story.append(Spacer(1, 0.2 * inch))
        
        # Sentiment
        sentiment_header = Paragraph("Overall Sentiment", self.styles['CustomHeader'])
        story.append(sentiment_header)
        sentiment_text = Paragraph(analysis['sentiment'], self.styles['CustomBody'])
        story.append(sentiment_text)
        
        story.append(PageBreak())
        return story
    
    def _create_participant_section(self, session_data: Dict[str, Any]) -> List[Any]:
        """Create participant statistics section"""
        story = []
        
        # Section header
        header = Paragraph("Participant Statistics", self.styles['CustomHeader'])
        story.append(header)
        story.append(Spacer(1, 0.2 * inch))
        
        # Participant table
        table_data = [['Participant', 'Speaking Time', 'Percentage']]
        
        for participant in session_data['participants']:
            table_data.append([
                participant['name'],
                participant['speaking_time'],
                f"{participant['percentage']}%"
            ])
        
        participant_table = Table(table_data, colWidths=[3*inch, 2*inch, 1.5*inch])
        participant_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), self.style.header_color),
            ('TEXTCOLOR', (0, 0), (-1, 0), black),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), self.style.font_family + '-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), self.style.font_size),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('BACKGROUND', (0, 1), (-1, -1), gray),
            ('GRID', (0, 0), (-1, -1), 1, self.style.table_border_color)
        ]))
        
        story.append(participant_table)
        story.append(PageBreak())
        
        return story
    
    def _create_appendix_section(self, session_data: Dict[str, Any]) -> List[Any]:
        """Create appendix section"""
        story = []
        
        # Section header
        header = Paragraph("Appendix", self.styles['CustomHeader'])
        story.append(header)
        story.append(Spacer(1, 0.2 * inch))
        
        # Technical information
        tech_info = [
            ['Generated by:', 'Silent Steno Device'],
            ['Generation date:', datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
            ['Session ID:', session_data['session_id']],
            ['Document version:', '1.0']
        ]
        
        tech_table = Table(tech_info, colWidths=[2*inch, 4*inch])
        tech_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('FONTNAME', (0, 0), (0, -1), self.style.font_family + '-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), self.style.font_size - 1),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ]))
        
        story.append(tech_table)
        
        return story
    
    def generate_transcript_pdf(self, session_id: str, output_path: Optional[str] = None) -> str:
        """Generate transcript-only PDF"""
        # Temporarily disable other sections
        original_config = self.config
        self.config = PDFConfig(
            include_cover_page=True,
            include_table_of_contents=False,
            include_transcript=True,
            include_analysis=False,
            include_participant_stats=False,
            include_appendix=False
        )
        
        try:
            result = self.generate_session_pdf(session_id, output_path)
            return result
        finally:
            self.config = original_config
    
    def generate_analysis_pdf(self, session_id: str, output_path: Optional[str] = None) -> str:
        """Generate analysis-only PDF"""
        # Temporarily disable other sections
        original_config = self.config
        self.config = PDFConfig(
            include_cover_page=True,
            include_table_of_contents=False,
            include_transcript=False,
            include_analysis=True,
            include_participant_stats=True,
            include_appendix=False
        )
        
        try:
            result = self.generate_session_pdf(session_id, output_path)
            return result
        finally:
            self.config = original_config

def create_pdf_generator(config: Optional[PDFConfig] = None, 
                        style: Optional[DocumentStyle] = None) -> PDFGenerator:
    """Create PDF generator with default or provided configuration"""
    if config is None:
        config = PDFConfig()
    
    return PDFGenerator(config, style)

def generate_session_pdf(session_id: str, output_path: Optional[str] = None) -> str:
    """Convenience function to generate session PDF"""
    generator = create_pdf_generator()
    return generator.generate_session_pdf(session_id, output_path)

def generate_transcript_pdf(session_id: str, output_path: Optional[str] = None) -> str:
    """Convenience function to generate transcript PDF"""
    generator = create_pdf_generator()
    return generator.generate_transcript_pdf(session_id, output_path)

def generate_analysis_pdf(session_id: str, output_path: Optional[str] = None) -> str:
    """Convenience function to generate analysis PDF"""
    generator = create_pdf_generator()
    return generator.generate_analysis_pdf(session_id, output_path)

def create_pdf_template(name: str, style: DocumentStyle, config: PDFConfig) -> PDFTemplate:
    """Create PDF template"""
    return PDFTemplate(
        name=name,
        description=f"PDF template: {name}",
        style=style,
        config=config
    )

def apply_document_style(generator: PDFGenerator, style: DocumentStyle):
    """Apply document style to PDF generator"""
    generator.style = style
    generator._customize_styles()