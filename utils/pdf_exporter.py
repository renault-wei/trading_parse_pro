import pdfkit
import pandas as pd
from pathlib import Path

class PDFExporter:
    """PDF 导出器"""
    
    def __init__(self, output_dir='trading_system/output/pdf_reports', wkhtmltopdf_path='D:/Program Files (x86)/wkhtmltopdf/bin/wkhtmltopdf.exe'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.wkhtmltopdf_path = wkhtmltopdf_path  # 替换为实际路径
    
    def export_to_pdf(self, html_content: str, report_title: str, filename: str):
        """将 HTML 内容导出为 PDF
        
        Args:
            html_content: 要导出的 HTML 内容
            report_title: 报告标题
            filename: 输出文件名
        """
        # 检查 HTML 内容
        if not isinstance(html_content, str) or not html_content.strip():
            raise ValueError("无效的 HTML 内容，无法导出为 PDF。")
        
        # 生成 PDF
        pdf_path = self.output_dir / f"{filename}.pdf"
        pdfkit.from_string(html_content, str(pdf_path), configuration=pdfkit.configuration(wkhtmltopdf=self.wkhtmltopdf_path))
        print(f"PDF 已生成: {pdf_path}")
    
    def _dataframe_to_html(self, data: pd.DataFrame, title: str) -> str:
        """将 DataFrame 转换为 HTML 格式"""
        html = f"<h1>{title}</h1>"
        html += data.to_html(index=False)
        return html 