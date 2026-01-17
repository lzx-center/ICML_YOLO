import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from enum import Enum
import re

def compute_expansions(
    df: pd.DataFrame,
    regions: Optional[List[Tuple[Tuple[int, int], Tuple[int, int]]]] = None
) -> List[List[Optional[Tuple[int, int]]]]:
    """
    给定 DataFrame(只考虑元素值),返回与其形状相同的二维列表。
    对每个未被覆盖的单元格 (i,j),优先向右扩展尽可能多相同元素得到宽 r,
    然后计算向下扩展的最大高度 d,要求每一行的这 r 列都与起点相等。
    起点记录为 (down, right) = (d, r),被其他矩形覆盖的单元格填 None。

    说明：返回的元组顺序已修改为 (向下拓展的大小, 向右拓展的大小) 以便与 (row, col) 坐标对应。
    示例：如果能向右扩展 3,向下扩展 5,则返回 (5, 3)。

    新增参数 regions: 可选列表,指定若干不相交的矩形区域,格式为
        (row_start, col_start, width, height)
    仅在这些区域内部执行拓展算法；区域外的单元格直接返回 (1, 1)。
    坐标采用 0-based。
    """
    vals = df.values
    rows, cols = vals.shape
    result: List[List[Optional[Tuple[int, int]]]] = [[None] * cols for _ in range(rows)]
    covered = [[False] * cols for _ in range(rows)]

    # 构建每个单元格所属的 region id(-1 表示不属于任何 region)
    region_idx = [[-1] * cols for _ in range(rows)]
    if regions:
        for ridx, ((r0, c0), (h, w)) in enumerate(regions):
            r_end = min(rows, r0 + h)
            c_end = min(cols, c0 + w)
            for ii in range(max(0, r0), r_end):
                for jj in range(max(0, c0), c_end):
                    region_idx[ii][jj] = ridx

    def eq(a: Any, b: Any) -> bool:
        # 把空字符串/NaN 视为等价(保持原有语义)
        if a == '' or b == '':
            return True
        if pd.isna(a) or pd.isna(b):
            return True
        return a == b

    # 对于不在任何 region 的单元格,直接返回 (1,1) 并标记为已处理
    if regions:
        for i in range(rows):
            for j in range(cols):
                if region_idx[i][j] == -1:
                    # (down, right) 对称为 (1,1)
                    result[i][j] = (1, 1)
                    covered[i][j] = True

    for i in range(rows):
        for j in range(cols):
            if covered[i][j]:
                continue
            start_region = region_idx[i][j]
            # 如果提供了 regions,但当前单元格不在任何 region,应已被标记为 covered
            # 优先向右扩展,但限制在同一 region 内
            base = vals[i, j]
            r = 1
            while j + r < cols and region_idx[i][j + r] == start_region and eq(vals[i, j + r], base) and (not covered[i][j + r]):
                r += 1
            # 然后向下扩展,要求每一新行上的 j..j+r-1 全部等于 base 且在同一 region
            d = 1
            while i + d < rows:
                ok = True
                for c in range(j, j + r):
                    if region_idx[i + d][c] != start_region or not eq(vals[i + d][c], base):
                        ok = False
                        break
                if ok:
                    d += 1
                else:
                    break
            # 记录起点并标记覆盖区域；被覆盖的单元格置 None
            # 注意：返回 (down, right)
            result[i][j] = (d, r)
            for ii in range(i, i + d):
                for jj in range(j, j + r):
                    covered[ii][jj] = True
                    if not (ii == i and jj == j):
                        result[ii][jj] = None
                    if ii == i + d - 1 and d > 1:
                        result[ii][jj] = 'b' # 最后一行的单元格不向下扩展
    return result

class LatexConverter:
    def __init__(self, dataframe: pd.DataFrame):
        self.dataframe = dataframe

    def _convert_heading(self, head_format=None, head_str=None, regions=None) -> str:
        # Convert LaTeX section headings to Markdown
        colums = self.dataframe.columns
        if isinstance(colums, pd.MultiIndex):
            width = len(colums)
            height = len(colums[0])
            data = pd.DataFrame([
                [colums[i][j] for i in range(width)]
                for j in range(height)
            ])
        else:
            data = pd.DataFrame([list(colums)])
        expansions = compute_expansions(data, regions=regions)
        # print("heading expansions:")
        # for row in expansions:
        #     print(row)
        def _escape_tex(x: Any) -> str:
            if x is None:
                return ''
            s = str(x)
            # 简单转义常见 LaTeX 特殊字符
            # for ch in ['\\', '&', '%', '$', '#', '_', '{', '}', '~', '^']:
            #     s = s.replace(ch, '\\' + ch)
            # return s
            return s
        rows, cols = data.shape

        latex_heading_str = ""
        if not head_format:
            head_format = ''
            for col in range(cols):
                if expansions[0][col] and expansions[0][col] != 'b':
                    d, r = expansions[0][col]
                    if d > 1 and r > 1:
                        head_format += 'c' * r
                        if col + r < cols - 1:
                            head_format += '|'
                    else:
                        for t in range(r):
                            head_format += 'c'
                            if col + t < cols - 1:
                                head_format += '|'
        latex_heading_str += f"{{{head_format}}}\n"
        # latex_heading_str += "\\toprule\n"
        latex_heading_str += "\\hline\n"


        for row in range(rows):
            col = 0
            while col < cols:
                exp = expansions[row][col]
                cell = data.iat[row, col]
                # 被覆盖的单元格：在当前行需要占位(保持列对齐),加入空占位符
                if exp is None or exp == 'b':
                    if col < cols - 1:
                        latex_heading_str += " & "
                    col += 1
                    continue


                # exp 格式为 (down, right)
                d, r = exp
                text = _escape_tex(cell)

                # 生成嵌套的 multirow/multicolumn 表达式
                multirow_format = 'c'
                if col > 0:
                    if col + r < cols:
                        multirow_format = 'c|'
                if d > 1 and r > 1:
                    fragment = f"\\multicolumn{{{r}}}{{{multirow_format}}}{{\\multirow{{{d}}}{{*}}{{{text}}}}}"
                elif d > 1:
                    fragment = f"\\multirow{{{d}}}{{*}}{{{text}}}"
                elif r > 1:
                    fragment = f"\\multicolumn{{{r}}}{{{multirow_format}}}{{{text}}}"
                else:
                    fragment = text

                latex_heading_str += fragment
                # 该单元格占据 r 列,若后面还有列则加分隔符
                if col + r < cols:
                    latex_heading_str += " & "
                col += r

            latex_heading_str += "\\\\"
        # add cline
            for col in range(cols):
                exp = expansions[row][col]
                if exp is None:
                    continue
                elif exp == 'b':
                    latex_heading_str += f"\\cline{{{col+1}-{col+1}}}"
                    continue
                d, r = exp
                if d > 1:
                    continue
                latex_heading_str += f"\\cline{{{col+1}-{col+r}}}"

            latex_heading_str += "\n"
        print(latex_heading_str)
        return latex_heading_str

    def _convert_body(self, regions=None) -> str:
        # Convert LaTeX table body to Markdown format
        body = self.dataframe.fillna('')
        expansions = compute_expansions(body, regions=regions)
        body = body.to_numpy()
        latex_body_str = ""
        rows, cols = body.shape
        for row in range(rows):
            for col in range(cols):
                if expansions[row][col]:
                    if not expansions[row][col] == 'b':
                        d, r = expansions[row][col]
                        if d > 1 and r > 1:
                            latex_body_str += f"\\multirow{{{d}}}{{*}}{{\\multicolumn{{{r}}}{{c}}{{{body[row, col]}}}}}"
                        elif d > 1:
                            latex_body_str += f"\\multirow{{{d}}}{{*}}{{{body[row, col]}}} "
                        elif r > 1:
                            latex_body_str += f"\\multicolumn{{{r}}}{{c}}{{{body[row, col]}}} "
                        else:
                            latex_body_str += f"{body[row, col]} "
                if col + r < cols:
                    latex_body_str += " & "
            latex_body_str = latex_body_str + "\\\\ \n"
            if row == rows - 1:
                # latex_body_str += "\\bottomrule\n"
                latex_body_str += "\\hline\n"
            else:
                for col in range(cols):
                    exp = expansions[row][col]
                    if exp is None:
                        continue
                    elif exp == 'b':
                        latex_body_str += f"\\cline{{{col+1}-{col+1}}}"
                        continue
                    d, r = exp
                    if d > 1:
                        continue
                    latex_body_str += f"\\cline{{{col+1}-{col+r}}}"
                latex_body_str += "\n"
        # print("body expansions:")
        # for row in expansions:  
        #     print(row)
        # print(latex_body_str)
        return latex_body_str



    def convert(self, head_region=None, body_region=None) -> str:
        latex_str = ""
        latex_str += "\\begin{tabular}\n"
        latex_str += self._convert_heading(regions=head_region)
        latex_str += self._convert_body(regions=body_region)
        latex_str += "\\end{tabular}\n"
        return latex_str

def test_multiindex():
    data = [
        ['A', 'A', 'B', 'B', 'B', 'C'],
        ['',  '',  'B1','B1', 'B2',''],
        ['',  '',  'B3','B4','', ''],
    ]
    mi = pd.MultiIndex.from_arrays(data)

    # 用 MultiIndex 作为列索引创建 DataFrame(两行示例)
    df = pd.DataFrame(
        [[1, 2, 3, 4, 5, 6],
         [6, 7, 8, 9, 10, 11]],
        columns=mi
    )
    print("初始 DataFrame:\n", df, "\n")

    # 添加新行(使用 loc；索引用当前长度 len(df))
    # 未给新列值的列会被填为 NaN
    df.loc[len(df)] = [11, 12, 13, 14, 15, 16]
    print("添加新行后:\n", df, "\n")

    # 添加单个新列(MultiIndex 列名用元组指定)
    df[('D', 'new', 'new')] = [100, 200, 300]  # 列长度需与行数一致(此处行数为3)
    df = df.reset_index(drop=True)  # 重置索引以避免潜在问题
    print("添加单列后:\n", df, "\n")
    print(df.columns)

    converter = LatexConverter(df)
    latex = converter.convert()
    print(latex)

def test_singleindex():
    data = {
        'A': [1, 1, 3],
        'B': [1, 1, 6],
        'C': [7, 8, 9]
    }
    df = pd.DataFrame(data)
    converter = LatexConverter(df)
    converter._convert_heading()
    converter._convert_body()
if __name__ == "__main__":
    test_multiindex()
    # test_singleindex()
