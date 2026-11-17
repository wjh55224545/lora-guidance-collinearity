# -*- coding: utf-8 -*-
"""docx -> PDF（必须用 D:\\Anaconda3\\python.exe，只有它有 pywin32）。

用法：python _doc2pdf.py <a.docx> [b.docx ...]
SaveAs(..., 17) 会报「命令失败」，只能用 ExportAsFixedFormat；路径含空格时
必须用具名参数，否则会被 URL 编码成 %20 导致找不到文件。
"""
import sys
import os
import win32com.client as win32

word = win32.Dispatch("Word.Application")
word.Visible = False
word.DisplayAlerts = 0
for p in sys.argv[1:]:
    p = os.path.normpath(p)
    out = os.path.splitext(p)[0] + ".pdf"
    d = word.Documents.Open(FileName=p, ConfirmConversions=False,
                            ReadOnly=False, AddToRecentFiles=False)
    if os.path.exists(out):
        os.remove(out)
    d.ExportAsFixedFormat(OutputFileName=out, ExportFormat=17)
    d.Close(False)
    print("->", out, os.path.getsize(out) // 1024, "KB")
word.Quit()
