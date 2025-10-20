#!/usr/bin/env python3
"""
Script to generate PDFs from markdown documentation files.
Creates professional PDF reports for the AI Knowledge Search Platform.
"""

import os
import sys
from pathlib import Path

def create_pdf_from_markdown(markdown_file, output_file):
    """Create PDF from markdown file using markdown2pdf."""
    try:
        # Import markdown2pdf
        from markdown2pdf import convert_markdown_to_pdf
        
        # Read markdown content
        with open(markdown_file, 'r', encoding='utf-8') as f:
            markdown_content = f.read()
        
        # Convert to PDF
        convert_markdown_to_pdf(markdown_content, output_file)
        
        print(f"✅ Created PDF: {output_file}")
        return True
        
    except ImportError:
        print("❌ markdown2pdf not available, trying alternative method...")
        return create_pdf_alternative(markdown_file, output_file)
    except Exception as e:
        print(f"❌ Error creating PDF: {str(e)}")
        return False

def create_pdf_alternative(markdown_file, output_file):
    """Alternative PDF creation using weasyprint."""
    try:
        import markdown2
        from weasyprint import HTML, CSS
        from weasyprint.text.fonts import FontConfiguration
        
        # Read markdown content
        with open(markdown_file, 'r', encoding='utf-8') as f:
            markdown_content = f.read()
        
        # Convert markdown to HTML
        html_content = markdown2.markdown(markdown_content, extras=['tables', 'fenced-code-blocks'])
        
        # Add CSS styling
        css_content = """
        @page {
            size: A4;
            margin: 2cm;
        }
        
        body {
            font-family: 'Helvetica', 'Arial', sans-serif;
            line-height: 1.6;
            color: #333;
        }
        
        h1 {
            color: #1f77b4;
            border-bottom: 2px solid #1f77b4;
            padding-bottom: 10px;
        }
        
        h2 {
            color: #2c3e50;
            margin-top: 30px;
        }
        
        h3 {
            color: #34495e;
        }
        
        code {
            background-color: #f8f9fa;
            padding: 2px 4px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
        }
        
        pre {
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
        }
        
        blockquote {
            border-left: 4px solid #1f77b4;
            margin: 0;
            padding-left: 20px;
            color: #666;
        }
        
        table {
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }
        
        th, td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }
        
        th {
            background-color: #f2f2f2;
        }
        """
        
        # Create full HTML document
        full_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>AI Knowledge Search Platform</title>
        </head>
        <body>
            {html_content}
        </body>
        </html>
        """
        
        # Generate PDF
        font_config = FontConfiguration()
        html_doc = HTML(string=full_html)
        css_doc = CSS(string=css_content, font_config=font_config)
        
        html_doc.write_pdf(output_file, stylesheets=[css_doc], font_config=font_config)
        
        print(f"✅ Created PDF: {output_file}")
        return True
        
    except Exception as e:
        print(f"❌ Error creating PDF with alternative method: {str(e)}")
        return False

def main():
    """Main function to generate all PDFs."""
    print("📄 Generating PDFs for AI Knowledge Search Platform")
    print("=" * 60)
    
    # Define files to convert
    files_to_convert = [
        ("PROJECT_REPORT.md", "AI_Knowledge_Search_Platform_Report.pdf"),
        ("ONE_PAGE_EXPLAINER.md", "AI_Knowledge_Search_Platform_Explainer.pdf"),
        ("README.md", "AI_Knowledge_Search_Platform_README.pdf"),
        ("ARCHITECTURE.md", "AI_Knowledge_Search_Platform_Architecture.pdf"),
        ("demo/DEMO_SCRIPT.md", "AI_Knowledge_Search_Platform_Demo_Script.pdf")
    ]
    
    success_count = 0
    total_count = len(files_to_convert)
    
    for markdown_file, pdf_file in files_to_convert:
        if os.path.exists(markdown_file):
            print(f"\n📝 Converting: {markdown_file} → {pdf_file}")
            if create_pdf_from_markdown(markdown_file, pdf_file):
                success_count += 1
        else:
            print(f"⚠️  File not found: {markdown_file}")
    
    print(f"\n📊 PDF Generation Summary:")
    print(f"✅ Successfully created: {success_count}/{total_count} PDFs")
    
    if success_count > 0:
        print(f"\n📁 Generated PDFs:")
        for _, pdf_file in files_to_convert:
            if os.path.exists(pdf_file):
                print(f"   - {pdf_file}")
    
    print(f"\n🎉 PDF generation complete!")
    return success_count == total_count

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
