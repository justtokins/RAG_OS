"""
Formatter module for converting raw LLM output to human-readable display.
"""
import re
from typing import Optional


class OutputFormatter:
    """Format LLM and tool outputs for human readability."""
    
    @staticmethod
    def format_answer(raw_answer: str, style: str = "markdown") -> str:
        """
        Format a raw LLM answer into human-readable output.
        
        Parameters
        ----------
        raw_answer : str
            The raw LLM response
        style : str
            Output style: 'markdown', 'plain', or 'html'
        
        Returns
        -------
        str
            Formatted answer
        """
        if style == "markdown":
            return OutputFormatter._format_markdown(raw_answer)
        elif style == "html":
            return OutputFormatter._format_html(raw_answer)
        else:
            return OutputFormatter._format_plain(raw_answer)
    
    @staticmethod
    def _format_markdown(text: str) -> str:
        """Format for Markdown output."""
        # Ensure proper spacing around sections
        text = re.sub(r'\n\n+', '\n\n', text)
        
        # Enhance code blocks
        text = re.sub(
            r'```(\w+)?\n',
            r'```\1\n',
            text
        )
        
        # Ensure headers have proper spacing
        text = re.sub(r'\n(#{1,6}\s)', r'\n\n\1', text)
        
        return text.strip()
    
    @staticmethod
    def _format_html(text: str) -> str:
        """Format for HTML output."""
        # Escape HTML
        text = (
            text
            .replace('&', '&amp;')
            .replace('<', '&lt;')
            .replace('>', '&gt;')
        )
        
        # Convert markdown-style headers to HTML
        text = re.sub(r'^### (.*?)$', r'<h3>\1</h3>', text, flags=re.MULTILINE)
        text = re.sub(r'^## (.*?)$', r'<h2>\1</h2>', text, flags=re.MULTILINE)
        text = re.sub(r'^# (.*?)$', r'<h1>\1</h1>', text, flags=re.MULTILINE)
        
        # Convert line breaks
        text = text.replace('\n\n', '</p><p>')
        text = f'<p>{text}</p>'
        
        return text
    
    @staticmethod
    def _format_plain(text: str) -> str:
        """Format for plain text output."""
        # Simple plain text cleanup
        text = re.sub(r'\n\n+', '\n\n', text)
        return text.strip()
    
    @staticmethod
    def format_context(context: str, max_length: int = 500) -> str:
        """
        Format retrieved context for display.
        
        Parameters
        ----------
        context : str
            Raw context from retrieval
        max_length : int
            Maximum characters to show
        
        Returns
        -------
        str
            Formatted context
        """
        if len(context) > max_length:
            context = context[:max_length] + "...\n[Context truncated]"
        
        # Add markers
        context = f"\n📚 **Context:**\n{context}\n"
        return context
    
    @staticmethod
    def format_error(error_msg: str, error_type: str = "error") -> str:
        """
        Format error messages for display.
        
        Parameters
        ----------
        error_msg : str
            The error message
        error_type : str
            Type of error ('error', 'warning', 'info')
        
        Returns
        -------
        str
            Formatted error
        """
        icons = {
            'error': '❌',
            'warning': '⚠️',
            'info': 'ℹ️',
        }
        icon = icons.get(error_type, '❓')
        return f"{icon} **{error_type.upper()}**: {error_msg}"
    
    @staticmethod
    def format_list(items: list[str], numbered: bool = False) -> str:
        """
        Format a list of items.
        
        Parameters
        ----------
        items : list[str]
            Items to format
        numbered : bool
            Use numbered list vs bullet points
        
        Returns
        -------
        str
            Formatted list
        """
        if numbered:
            formatted = '\n'.join(f"{i+1}. {item}" for i, item in enumerate(items))
        else:
            formatted = '\n'.join(f"• {item}" for item in items)
        return formatted
    
    @staticmethod
    def format_table(headers: list[str], rows: list[list[str]]) -> str:
        """
        Format data as a markdown table.
        
        Parameters
        ----------
        headers : list[str]
            Column headers
        rows : list[list[str]]
            Table rows
        
        Returns
        -------
        str
            Formatted table
        """
        # Calculate column widths
        col_widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                col_widths[i] = max(col_widths[i], len(str(cell)))
        
        # Build table
        lines = []
        
        # Header
        header_row = " | ".join(
            h.ljust(col_widths[i]) for i, h in enumerate(headers)
        )
        lines.append(header_row)
        
        # Separator
        sep = " | ".join("-" * w for w in col_widths)
        lines.append(sep)
        
        # Rows
        for row in rows:
            data_row = " | ".join(
                str(cell).ljust(col_widths[i]) for i, cell in enumerate(row)
            )
            lines.append(data_row)
        
        return "\n".join(lines)
