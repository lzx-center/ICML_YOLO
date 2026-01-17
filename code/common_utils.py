import pandas as pd
import pandas as pd
import os # Assuming os is needed if you uncomment the file operations later
import numpy as np
from typing import List, Optional # Added for type hinting


def escape_latex(text_val, from_col=None) -> str:
    """Escapes special LaTeX characters in a string."""
    if pd.isna(text_val):
        return ""
    if text_val == np.inf:
        return "0.00"
    if isinstance(text_val, float) or isinstance(text_val, int) or isinstance(text_val,np.float64) or isinstance(text_val,np.int64) or isinstance(text_val, np.int32) or isinstance(text_val, np.float32):
        if from_col in {'verified', r'${\tau=0.5}$', r'${\tau=0.7}$', '0.5 delta pgd', '0.7 delta pgd'}:
            text = f"{text_val:.2f}"
        else:
            text = f"{text_val:.1f}" # Scientific notation
    else:
        text = str(text_val)
    
    # Order is important: replace backslash first, then other characters.
    text = text.replace('&', '\\&').replace('%', '\\%') \
               .replace('#', '\\#') \
               .replace('~', '\\textasciitilde{}') \
               .replace('^', '\\textasciicircum{}')
    return text

def df_to_latex_multirow(df: pd.DataFrame) -> str:
    """
    Converts a CSV file to a LaTeX table string with multirow support for merged cells
    and appropriate horizontal lines using \cline.
    """

    num_rows, num_cols = df.shape
    if num_rows == 0:
        # Create a header line even for an empty data table
        if num_cols > 0:
            header_escaped = [escape_latex(col) for col in df.columns]
            header_str = " & ".join(header_escaped) + " \\\\"
            col_format_str = "l" * num_cols # Default to left alignment
            return f"\\begin{{tabular}}{{{col_format_str}}}\n\\hline\n{header_str}\n\\hline\n\\end{{tabular}}"
        else:
            return "\\begin{tabular}{}\n\\hline\n\\hline\n\\end{tabular}"


    output_grid = [[escape_latex(df.iloc[r, c]) for c in range(num_cols)] for r in range(num_rows)]
    active_multirow_end_row = [-1] * num_cols

    for r in range(num_rows):
        for c in range(num_cols):
            if r <= active_multirow_end_row[c]:
                output_grid[r][c] = "" 
                continue

            current_val = df.iloc[r, c]
            span = 1
            for k in range(r + 1, num_rows):
                if df.iloc[k, c] != current_val:
                    break
                
                prev_cols_aligned = True
                for prev_c in range(c):
                    if df.iloc[k, prev_c] != df.iloc[r, prev_c]:
                        prev_cols_aligned = False
                        break
                
                if not prev_cols_aligned:
                    break
                span += 1
            
            if span > 1:
                output_grid[r][c] = f"\\multirow{{{span}}}{{*}}{{{escape_latex(current_val)}}}"
                active_multirow_end_row[c] = r + span - 1

    header_escaped = [escape_latex(col) for col in df.columns]
    header_str = " & ".join(header_escaped) + " \\\\"
    
    col_formats = []
    for col_idx in range(num_cols):
        is_numeric = False
        try:
            # Attempt to convert to numeric, dropna to handle potential mixed types gracefully for check
            pd.to_numeric(df.iloc[:, col_idx].dropna())
            # Check if original non-NaN values were all numeric (handles cases like object dtype with numbers)
            if df.iloc[:, col_idx].dropna().apply(lambda x: isinstance(x, (int, float))).all():
                 is_numeric = True
        except ValueError:
            is_numeric = False
            
        if is_numeric:
            col_formats.append('r')
        else:
            col_formats.append('l')
    col_format_str = "".join(col_formats)

    latex_string = f"\\begin{{tabular}}{{{col_format_str}}}\n"
    latex_string += "\\hline\n" 
    latex_string += header_str + "\n"
    latex_string += "\\hline\n" 

    for r in range(num_rows):
        if r > 0:
            needs_cline = False
            first_diff_col_idx = -1 # 0-indexed
            # Find the first column where the current row differs from the previous one,
            # indicating a break in a merged block.
            for c_check in range(num_cols):
                if df.iloc[r, c_check] != df.iloc[r-1, c_check]:
                    # Ensure all preceding columns were identical to confirm hierarchical break
                    is_hierarchical_break = True
                    for prev_c in range(c_check):
                        if df.iloc[r, prev_c] != df.iloc[r-1, prev_c]:
                            is_hierarchical_break = False
                            break
                    if is_hierarchical_break:
                        needs_cline = True
                        first_diff_col_idx = c_check
                        break
            
            if needs_cline:
                # \cline should start from the first differing column (1-indexed)
                # and span to the last column.
                latex_string += f"\\cline{{{first_diff_col_idx + 1}-{num_cols}}}\n"

        row_content_str = " & ".join(output_grid[r]) + " \\\\"
        latex_string += row_content_str + "\n"

    latex_string += "\\hline\n"
    latex_string += "\\end{tabular}\n"
    return latex_string



def df_to_latex_multirow2(df: pd.DataFrame, columns_to_merge: Optional[List[str]] = None, header_str = None, col_format_str = None) -> str:
    """
    Converts a DataFrame to a LaTeX table string with multirow support for merged cells
    in specified columns and appropriate horizontal lines using \cline.
    If columns_to_merge is None, attempts to merge all columns hierarchically.
    Otherwise, only columns named in columns_to_merge will receive \multirow commands.
    Merged columns will not have internal horizontal lines from \cline.
    """

    num_rows, num_cols = df.shape
    if num_rows == 0:
        if num_cols > 0:
            header_escaped = [escape_latex(col, df.columns[c]) for col in df.columns]
            header_str = " & ".join(header_escaped) + " \\\\"
            col_format_str = "l" * num_cols
            return f"\\begin{{tabular}}{{{col_format_str}}}\n\\hline\n{header_str}\n\\hline\n\\end{{tabular}}"
        else:
            return "\\begin{tabular}{}\n\\hline\n\\hline\n\\end{tabular}"

    output_grid = [[escape_latex(df.iloc[r, c], df.columns[c]) for c in range(num_cols)] for r in range(num_rows)]
    active_multirow_end_row = [-1] * num_cols
    is_blanked_by_multirow = [[False for _ in range(num_cols)] for _ in range(num_rows)]

    eligible_for_multirow_command = [False] * num_cols
    if columns_to_merge is None:
        eligible_for_multirow_command = [True] * num_cols
    else:
        for c_idx, col_name in enumerate(df.columns):
            if col_name in columns_to_merge:
                eligible_for_multirow_command[c_idx] = True

    for r in range(num_rows):
        for c in range(num_cols):
            if r <= active_multirow_end_row[c]: 
                output_grid[r][c] = ""
                is_blanked_by_multirow[r][c] = True # Mark as blanked due to multirow
                continue

            current_val = df.iloc[r, c]
            span = 1
            for k in range(r + 1, num_rows):
                if df.iloc[k, c] != current_val:
                    break
                
                apply_hierarchical_constraint = True
                if columns_to_merge is not None and eligible_for_multirow_command[c]:
                    apply_hierarchical_constraint = False
                
                if apply_hierarchical_constraint:
                    prev_cols_aligned = True
                    for prev_c in range(c):
                        if df.iloc[k, prev_c] != df.iloc[r, prev_c]:
                            prev_cols_aligned = False
                            break
                    if not prev_cols_aligned:
                        break
                span += 1
            
            if span > 1 and eligible_for_multirow_command[c]:
                output_grid[r][c] = f"\\multirow{{{span}}}{{*}}{{{escape_latex(current_val)}}}"
                active_multirow_end_row[c] = r + span - 1
            # else: output_grid[r][c] already has escape_latex(current_val) from initialization

    header_escaped = [escape_latex(col) for col in df.columns]
    if header_str is None:
        header_str = " & ".join(header_escaped) + " \\\\"
    
    col_formats = []
    for col_idx in range(num_cols):
        is_numeric = False
        try:
            if pd.to_numeric(df.iloc[:, col_idx].dropna(), errors='raise').dtype in [np.int64, np.float64, np.int32, np.float32]:
                 is_numeric = True
            # Further check if all non-NaN were numeric if original type was object
            elif df.iloc[:, col_idx].dropna().apply(lambda x: isinstance(x, (int, float))).all() and not df.iloc[:, col_idx].dropna().empty:
                is_numeric = True
        except (ValueError, TypeError):
            is_numeric = False
            
        if is_numeric:
            col_formats.append('r')
        else:
            col_formats.append('l')
    if col_format_str is None:
        col_format_str = "".join(col_formats)

    latex_string = f"\\begin{{tabular}}{{{col_format_str}}}\n"
    latex_string += "\\hline\n" 
    latex_string += header_str + "\n"
    latex_string += "\\hline\n" 

    for r in range(num_rows):
        if r > 0:
            first_hierarchical_diff_col_idx = -1
            for c_check in range(num_cols):
                if df.iloc[r, c_check] != df.iloc[r-1, c_check]:
                    is_true_hierarchical_break = True
                    for prev_c in range(c_check):
                        if df.iloc[r, prev_c] != df.iloc[r-1, prev_c]:
                            is_true_hierarchical_break = False
                            break
                    if is_true_hierarchical_break:
                        first_hierarchical_diff_col_idx = c_check
                        break
            
            if first_hierarchical_diff_col_idx != -1:
                cline_segments = []
                current_segment_start = -1
                for c_segment in range(first_hierarchical_diff_col_idx, num_cols):
                    # A cell (r, c_segment) is protected if it's blanked by a multirow from above
                    is_protected = is_blanked_by_multirow[r][c_segment]

                    if not is_protected:
                        if current_segment_start == -1:
                            current_segment_start = c_segment
                    else: # is_protected
                        if current_segment_start != -1:
                            # End current segment before this protected column
                            cline_segments.append(f"\\cline{{{current_segment_start + 1}-{c_segment}}}")
                            current_segment_start = -1
                
                if current_segment_start != -1:
                    # Last segment goes to the end of the table
                    cline_segments.append(f"\\cline{{{current_segment_start + 1}-{num_cols}}}")
                
                if cline_segments:
                    latex_string += "".join(cline_segments) + "\n"

        row_content_str = " & ".join(output_grid[r]) + " \\\\"
        latex_string += row_content_str + "\n"

    latex_string += "\\hline\n"
    latex_string += "\\end{tabular}\n"
    return latex_string

def change_to_2_col(df):
    df_original = df.copy()
    num_original_rows = len(df_original)
    mid_point = num_original_rows // 2

    # First half of the rows, reset index for proper concatenation
    df_left = df_original.iloc[:mid_point].reset_index(drop=True)
    
    # Second half of the rows, reset index for proper concatenation
    df_right = df_original.iloc[mid_point:].reset_index(drop=True)
    
    # Concatenate the two halves side-by-side
    # If the original DataFrame had an odd number of rows, 
    # df_right will have one more row than df_left.
    # pd.concat(axis=1) will align them by index, and the resulting
    # DataFrame will have max(len(df_left), len(df_right)) rows,
    # with NaN values where data is missing in the shorter part.
    # Column names from df_original will be repeated.
    df_transformed = pd.concat([df_left, df_right], axis=1)
    return df_transformed