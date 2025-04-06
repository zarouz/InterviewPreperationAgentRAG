# report_generator.py
import logging
import datetime
import re

# --- ReportLab Imports ---
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, KeepTogether
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT, TA_CENTER
    from reportlab.lib.units import inch
except ImportError:
    logging.getLogger(__name__).critical("ReportLab library not found. Please install it: pip install reportlab")
    # Define dummy classes/functions if ReportLab is missing
    class SimpleDocTemplate: pass
    class Paragraph: pass
    class Spacer: pass
    class PageBreak: pass
    class Table: pass
    class TableStyle: pass
    class KeepTogether: pass
    def getSampleStyleSheet(): return {'Normal': None, 'h1': None, 'h2': None, 'h3': None, 'Italic': None, 'Code': None}
    class ParagraphStyle: pass
    letter = None; inch = None; colors = None; TA_LEFT = None; TA_CENTER = None

# --- Local Imports ---
import utils # Still needed for skill extraction
# import confidence_analyzer # <-- REMOVED IMPORT (Rating comes from API data)
from config import CANDIDATE_NAME, REPORT_FILENAME

# --- Logger ---
logger = logging.getLogger(__name__)

# --- ReportLab Styles Setup ---
# (Styles remain the same)
styles = {}
title_style = None
h2_style = None
h3_style = None
normal_style = None
italic_style = None
code_style = None
bullet_style = None
confidence_style = None
normal_indented_style = None # Added definition

if 'Normal' in getSampleStyleSheet(): # Check if ReportLab loaded successfully
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(name='TitleStyle', parent=styles['h1'], alignment=TA_CENTER, spaceAfter=16)
    h2_style = ParagraphStyle(name='H2Style', parent=styles['h2'], spaceBefore=12, spaceAfter=6)
    h3_style = ParagraphStyle(name='H3Style', parent=styles['h3'], spaceBefore=10, spaceAfter=4, leftIndent=10)
    normal_style = styles['Normal']
    italic_style = styles['Italic']
    code_style = styles.get('Code', ParagraphStyle(name='CodeStyle', parent=styles['Normal'], fontName='Courier', fontSize=9, leading=11))
    bullet_style = ParagraphStyle(name='BulletStyle', parent=styles['Normal'], leftIndent=30, firstLineIndent=0, spaceBefore=2)
    confidence_style = ParagraphStyle(name='ConfidenceStyle', parent=styles['Normal'], textColor=colors.darkblue if colors else "#00008B", leftIndent=18)
    normal_indented_style = ParagraphStyle(name='NormalIndented', parent=styles['Normal'], leftIndent=18)

# --- Report Generation Function ---
def generate_pdf_report(evaluated_data, resume_text, jd_text, role_title, report_filename=None):
    """
    Generates a PDF report summarizing the interview, evaluations, and confidence.
    NOTE: Assumes 'evaluated_data' contains confidence results fetched from the API.
    """

    try:
        from reportlab.platypus import SimpleDocTemplate
        if not normal_style: raise ImportError("ReportLab styles not initialized.")
    except ImportError:
        logger.error("ReportLab library not found or failed to initialize. Cannot generate PDF.")
        raise ImportError("ReportLab not installed or initialized correctly. Please run: pip install reportlab")

    if report_filename is None:
        report_filename = REPORT_FILENAME
    logger.info(f"Generating PDF report: {report_filename}")

    doc = SimpleDocTemplate(report_filename, pagesize=letter,
                            leftMargin=0.75*inch, rightMargin=0.75*inch,
                            topMargin=0.75*inch, bottomMargin=0.75*inch)
    story = []

    # --- Title Page ---
    # (Unchanged)
    story.append(Paragraph(f"Interview Evaluation Report", title_style))
    story.append(Spacer(1, 0.3*inch))
    candidate_name_display = CANDIDATE_NAME if CANDIDATE_NAME else "Candidate"
    story.append(Paragraph(f"Candidate: {candidate_name_display}", styles['h2']))
    story.append(Paragraph(f"Role: {role_title}", styles['h2']))
    story.append(Paragraph(f"Date: {datetime.date.today().strftime('%Y-%m-%d')}", normal_style))
    story.append(PageBreak())

    # --- Resume vs JD Analysis ---
    # (Unchanged)
    story.append(Paragraph("Resume vs. Job Description Skill Analysis", h2_style))
    story.append(Spacer(1, 0.1*inch))
    try:
        resume_skills = utils.extract_skills_from_text(resume_text)
        jd_skills = utils.extract_skills_from_text(jd_text)
        intersection = resume_skills.intersection(jd_skills)
        resume_only = resume_skills - jd_skills
        jd_only = jd_skills - resume_skills

        def format_skill_list_pdf(skill_set):
            if not skill_set: return Paragraph("<i>None identified in predefined list.</i>", italic_style)
            sorted_list = sorted(list(skill_set), key=lambda x: (-len(x.split()), x))
            return Paragraph(", ".join(sorted_list), normal_style)

        analysis_data = [
            [Paragraph('<b>Category</b>', normal_style), Paragraph('<b>Identified Skills / Keywords</b>', normal_style)],
            [Paragraph('Matching Skills:', normal_style), format_skill_list_pdf(intersection)],
            [Paragraph('Skills in JD (Potential Gaps):', normal_style), format_skill_list_pdf(jd_only)],
            [Paragraph('Skills in Resume (Not in JD):', normal_style), format_skill_list_pdf(resume_only)],
        ]
        table_style_def = TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue if colors else "#00008B"),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke if colors else "#FFFFFF"),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
            ('TOPPADDING', (0, 0), (-1, 0), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.black if colors else "#000000"),
            ('VALIGN', (0, 1), (-1, -1), 'TOP'),
            ('LEFTPADDING', (0, 0), (-1, -1), 6),
            ('RIGHTPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 1), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
        ])
        analysis_table = Table(analysis_data, colWidths=[1.8*inch, 5.2*inch])
        analysis_table.setStyle(table_style_def)
        story.append(analysis_table)
        story.append(Spacer(1, 0.2*inch))

    except Exception as e:
        logger.error(f"Error generating skill analysis section: {e}", exc_info=True)
        story.append(Paragraph("<i>Error generating skill analysis.</i>", italic_style))

    story.append(PageBreak())

    # --- Individual Question Evaluations ---
    story.append(Paragraph("Interview Question Evaluations", h2_style))
    story.append(Spacer(1, 0.1*inch))

    total_text_score = 0
    num_text_evaluated = 0
    total_confidence_score = 0
    num_confidence_evaluated = 0
    evaluation_summary = {'strengths': [], 'areas_for_improvement': []}

    for i, data in enumerate(evaluated_data):
        q_num = data.get('question_index', i + 1)
        question = data.get('question', 'N/A')
        response_text = data.get('response', '[No Text Response Recorded]')
        evaluation_text = data.get('evaluation', 'Evaluation not available.')
        text_score = data.get('score') # Score from LLM eval of text

        # Get confidence data directly from the 'data' dictionary (fetched from API)
        confidence_score = data.get('confidence_score')
        confidence_rating = data.get('confidence_rating', 'N/A') # Get rating string directly
        primary_emotion = data.get('primary_emotion', 'N/A')
        confidence_error = data.get('confidence_analysis_error', True)

        q_story = []
        q_story.append(Paragraph(f"Question {q_num}: {question}", h3_style))

        q_story.append(Paragraph(f"<i>Candidate Response (Transcribed):</i>", italic_style))
        q_story.append(Paragraph(response_text, code_style))
        q_story.append(Spacer(1, 0.1*inch))

        # --- Confidence Analysis Section (Data from API) ---
        q_story.append(Paragraph("<b>Voice Confidence Analysis:</b>", normal_style))
        if confidence_error:
             message = data.get('message', 'Analysis failed or not performed.') # Get msg if available
             q_story.append(Paragraph(f"<i>  {message}</i>", italic_style))
        else:
             conf_score_str = f"{confidence_score:.1f}%" if confidence_score is not None else "N/A"
             q_story.append(Paragraph(f"• Overall Confidence Score: <b>{conf_score_str}</b>", confidence_style))
             q_story.append(Paragraph(f"• Confidence Rating: <b>{confidence_rating}</b>", confidence_style)) # Use rating directly
             q_story.append(Paragraph(f"• Dominant Emotion (Aggregated): {primary_emotion}", normal_indented_style))
             if confidence_score is not None:
                 total_confidence_score += confidence_score
                 num_confidence_evaluated += 1
        q_story.append(Spacer(1, 0.1*inch))

        # --- Text Evaluation Section (Unchanged Logic) ---
        q_story.append(Paragraph("<b>Text Response Evaluation:</b>", normal_style))
        current_section = None
        if evaluation_text == 'Evaluation not available.' or evaluation_text.startswith("Evaluation Error:"):
             q_story.append(Paragraph(f"<i>  {evaluation_text}</i>", italic_style))
        else:
             evaluation_lines = evaluation_text.split('\n')
             for line_num, line in enumerate(evaluation_lines):
                 line_strip = line.strip()
                 if not line_strip: continue

                 match = re.match(r"^\*?\s*([\w\s/()\-]+?):\s*(.*)", line_strip, re.IGNORECASE)
                 bullet_match = re.match(r"^\s*[-*•]\s+(.*)", line_strip)

                 if match:
                     section_title_raw = match.group(1).strip()
                     section_title_lower = section_title_raw.lower()
                     section_content = match.group(2).strip()
                     q_story.append(Paragraph(f"• <b>{section_title_raw}:</b> {section_content}", normal_indented_style))
                     current_section = section_title_lower

                     if "strengths" in section_title_lower:
                         evaluation_summary['strengths'].append(f"Q{q_num}: {section_content}")
                     elif "improvement" in section_title_lower:
                         evaluation_summary['areas_for_improvement'].append(f"Q{q_num}: {section_content}")

                 elif bullet_match:
                      bullet_text = bullet_match.group(1).strip()
                      if current_section == "strengths" and evaluation_summary['strengths']:
                           if evaluation_summary['strengths'][-1].startswith(f"Q{q_num}:"):
                               evaluation_summary['strengths'][-1] += f"; {bullet_text}"
                           else:
                                evaluation_summary['strengths'].append(f"Q{q_num} (bullet): {bullet_text}")
                      elif current_section == "improvement" and evaluation_summary['areas_for_improvement']:
                           if evaluation_summary['areas_for_improvement'][-1].startswith(f"Q{q_num}:"):
                               evaluation_summary['areas_for_improvement'][-1] += f"; {bullet_text}"
                           else:
                                evaluation_summary['areas_for_improvement'].append(f"Q{q_num} (bullet): {bullet_text}")
                      q_story.append(Paragraph(f"- {bullet_text}", bullet_style))
                 else:
                      q_story.append(Paragraph(f"  {line_strip}", normal_indented_style))

        if text_score is not None:
            total_text_score += text_score
            num_text_evaluated += 1

        q_story.append(Spacer(1, 0.2*inch))
        story.append(KeepTogether(q_story))

    # --- Overall Summary Section ---
    story.append(PageBreak())
    story.append(Paragraph("Overall Assessment Summary", h2_style))
    story.append(Spacer(1, 0.1*inch))

    # --- Overall Text Score (Unchanged) ---
    story.append(Paragraph("<b>Text Evaluation Summary:</b>", h3_style))
    if num_text_evaluated > 0:
         average_text_score = total_text_score / num_text_evaluated
         story.append(Paragraph(f"• Average Text Score: {average_text_score:.1f} / 5.0 (based on {num_text_evaluated} evaluated responses)", normal_indented_style))
    else:
         story.append(Paragraph("• <i>No numerical text evaluation scores could be extracted.</i>", italic_style))
    story.append(Spacer(1, 0.1*inch))

    # Summarized Strengths (Unchanged)
    story.append(Paragraph("<u>Key Strengths Observed (from Text):</u>", normal_indented_style))
    if evaluation_summary['strengths']:
        for item in evaluation_summary['strengths']: story.append(Paragraph(item, bullet_style))
    else: story.append(Paragraph("<i>  - None explicitly listed in evaluations.</i>", italic_style))
    story.append(Spacer(1, 0.1*inch))

    # Summarized Areas for Improvement (Unchanged)
    story.append(Paragraph("<u>Potential Areas for Improvement (from Text):</u>", normal_indented_style))
    if evaluation_summary['areas_for_improvement']:
        for item in evaluation_summary['areas_for_improvement']: story.append(Paragraph(item, bullet_style))
    else: story.append(Paragraph("<i>  - None explicitly listed in evaluations.</i>", italic_style))
    story.append(Spacer(1, 0.2*inch))

    # --- Overall Confidence Score (Uses data from API) ---
    story.append(Paragraph("<b>Voice Confidence Summary:</b>", h3_style))
    if num_confidence_evaluated > 0:
        average_confidence_score = total_confidence_score / num_confidence_evaluated
        # --- Calculate average rating based on individual ratings from API ---
        # This requires ratings to be mapped to numerical values if you want an average rating string
        # Simpler: Just report the average score. If needed, calculate average rating based on score later.
        # Or, we can try to find the most frequent rating string.
        all_ratings = [d.get('confidence_rating') for d in evaluated_data if d.get('confidence_score') is not None and d.get('confidence_rating')]
        average_confidence_rating = "Mixed" # Default if calculation is complex
        if all_ratings:
             from collections import Counter
             rating_counts = Counter(all_ratings)
             most_common_rating = rating_counts.most_common(1)[0][0]
             average_confidence_rating = most_common_rating # Use most frequent as the 'average' label

        story.append(Paragraph(f"• Average Confidence Score: <b>{average_confidence_score:.1f}%</b>", confidence_style))
        # Report the most frequent rating or a simple note
        story.append(Paragraph(f"• Most Frequent Rating: <b>{average_confidence_rating}</b>", confidence_style))
        story.append(Paragraph(f"  (Based on {num_confidence_evaluated} successfully analyzed responses via API)", italic_style))
    else:
        story.append(Paragraph("• <i>No voice confidence scores were successfully retrieved from the API.</i>", italic_style))
    story.append(Spacer(1, 0.2*inch))

    story.append(Paragraph("<i>Note: This report combines AI evaluation of transcribed text and AI analysis of voice characteristics (via API). Review by a human interviewer is recommended. Confidence scores are indicative.</i>", italic_style))

    # --- Build PDF ---
    logger.info("Building PDF document...")
    try:
        doc.build(story)
        logger.info(f"Successfully generated report: {report_filename}")
    except Exception as e:
        logger.error(f"Error building PDF report with ReportLab: {e}", exc_info=True)
        raise