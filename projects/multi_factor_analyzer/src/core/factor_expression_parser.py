"""
因子表达式解析器模块

本模块提供因子表达式的解析和验证功能，主要用于检测因子计算中的未来函数。
未来函数是指在计算因子时使用了未来的数据，这会导致回测时的数据泄露问题。

主要功能：
1. 未来函数静态检测 - 检测因子表达式中的未来引用
2. 表达式验证 - 验证表达式的合法性
3. 安全性保证 - 确保因子计算只使用历史数据
"""

import re
from typing import List, Optional, Tuple


class FactorExpressionParser:
    """
    因子表达式解析器

    用于解析和验证因子表达式，检测其中可能存在的未来函数引用。
    未来函数是指在计算因子时使用了未来的数据，这会导致回测结果失真。

    检测原理：
    - 静态分析：通过正则表达式匹配已知的未来函数模式
    - 负索引检测：检测表达式中的负索引（如 $close[-1]）
    - Ref 函数检测：检测 Ref 函数的正向偏移量

    Attributes:
        FUTURE_PATTERNS: 禁止的未来函数模式列表

    Examples:
        >>> parser = FactorExpressionParser()
        >>>
        >>> # 验证安全的表达式
        >>> parser.validate_no_future_functions("Ref($close, 5)")
        True
        >>>
        >>> # 验证危险的表达式
        >>> try:
        ...     parser.validate_no_future_functions("Ref($close, -5)")
        ... except ValueError as e:
        ...     print(e)
        表达式包含未来函数: Ref($close, -5)
        >>>
        >>> # 验证自定义因子
        >>> expr = "$close / Ref($close, 20) - 1"
        >>> if parser.validate_no_future_functions(expr):
        ...     print("表达式安全")
        表达式安全
    """

    # 禁止的未来函数模式
    FUTURE_PATTERNS: List[str] = [
        r"Ref\s*\(\s*\$?\w+\s*,\s*-\s*\d+\s*\)",  # Ref($close, -N) where N>0
        r"\$\w+\[\s*-\s*\d+\s*\]",  # $close[-N] where N>0
        r"Roll\s*\(\s*\$?\w+\s*,\s*\d+\s*\)",  # Roll with positive offset
    ]

    def __init__(self, custom_patterns: Optional[List[str]] = None):
        """
        初始化因子表达式解析器

        Args:
            custom_patterns: 自定义的未来函数模式列表，会添加到默认模式中

        Examples:
            >>> parser = FactorExpressionParser(
            ...     custom_patterns=[r'Future\([^)]+\)']  # noqa: W605
            ... )
        """
        self.patterns = self.FUTURE_PATTERNS.copy()
        if custom_patterns:
            self.patterns.extend(custom_patterns)

    def validate_no_future_functions(self, expression: str) -> bool:
        """
        静态分析验证因子表达式不包含未来引用

        该方法会检查表达式是否包含已知的未来函数模式，包括：
        1. Ref 函数的正向偏移量（如 Ref($close, -5)）
        2. 负索引（如 $close[-1]）
        3. Roll 函数的正向偏移

        Args:
            expression: 因子表达式字符串

        Returns:
            bool: 如果表达式安全（无未来函数）返回 True

        Raises:
            ValueError: 如果检测到未来函数，抛出异常并包含详细说明

        Examples:
            >>> parser = FactorExpressionParser()
            >>>
            >>> # 安全的表达式
            >>> parser.validate_no_future_functions("Ref($close, 5)")
            True
            >>>
            >>> # 危险的表达式
            >>> try:
            ...     parser.validate_no_future_functions("Ref($close, -5)")
            ... except ValueError as e:
            ...     print(f"检测到错误: {e}")
            检测到错误: ...
        """
        # 检查所有未来函数模式
        for pattern in self.patterns:
            match = re.search(pattern, expression)
            if match:
                raise ValueError(
                    f"表达式包含未来函数: {expression}\n"
                    f"检测到模式: {pattern}\n"
                    f"匹配内容: {match.group()}\n"
                    "因子计算只能使用历史数据，不能引用未来数据。\n"
                    "请检查：\n"
                    "1. Ref 函数的第二个参数应该是正数\n"
                    "2. 避免使用负索引（如 $close[-1]）\n"
                    "3. 确保所有计算都基于历史数据"
                )

        # 检查是否有负索引（更广泛的检查）
        if self._has_negative_index(expression):
            raise ValueError(
                f"表达式包含负索引(未来引用): {expression}\n"
                "请确保所有索引都是非负整数。\n"
                "示例：\n"
                "  - 正确: Ref($close, 5) 或 $close[5]\n"
                "  - 错误: Ref($close, -5) 或 $close[-1]"
            )

        return True

    def _has_negative_index(self, expression: str) -> bool:
        """
        检测表达式中的负索引

        负索引是指类似 [ -N ] 或 [-N] 的模式，表示引用未来的数据。

        Args:
            expression: 要检查的表达式

        Returns:
            bool: 如果包含负索引返回 True

        Examples:
            >>> parser = FactorExpressionParser()
            >>> parser._has_negative_index("$close[5]")
            False
            >>> parser._has_negative_index("$close[-1]")
            True
        """
        # 查找类似 [ -N ] 或 [-N] 的模式
        # 这个模式会匹配：
        # - [-1]
        # - [ -5 ]
        # - [-10]
        negative_index_pattern = r"\[\s*-\s*\d+\s*\]"
        return bool(re.search(negative_index_pattern, expression))

    def check_expression_safety(self, expression: str) -> Tuple[bool, List[str]]:
        """
        检查表达式安全性，返回检查结果和警告列表

        这是一个更温和的检查方法，不会抛出异常，而是返回检查结果。

        Args:
            expression: 要检查的表达式

        Returns:
            Tuple[bool, List[str]]: (是否安全, 警告信息列表)

        Examples:
            >>> parser = FactorExpressionParser()
            >>> is_safe, warnings = parser.check_expression_safety("Ref($close, 5)")
            >>> print(is_safe)
            True
            >>> print(warnings)
            []
        """
        warnings = []

        # 检查未来函数
        for pattern in self.patterns:
            match = re.search(pattern, expression)
            if match:
                warnings.append(f"检测到未来函数: {match.group()} (模式: {pattern})")

        # 检查负索引
        if self._has_negative_index(expression):
            warnings.append("检测到负索引，可能存在未来引用")

        # 检查潜在的危险操作
        if "shift(" in expression.lower():
            warnings.append("检测到 shift 操作，请确保参数为负数或0")

        is_safe = len(warnings) == 0
        return is_safe, warnings

    def extract_fields(self, expression: str) -> List[str]:
        """
        从表达式中提取使用的字段

        提取所有以 $ 开头的字段名，如 $close, $volume 等。

        Args:
            expression: 因子表达式

        Returns:
            List[str]: 提取的字段列表

        Examples:
            >>> parser = FactorExpressionParser()
            >>> parser.extract_fields("$close / Ref($volume, 5)")
            ['close', 'volume']
        """
        # 匹配 $field_name 模式
        pattern = r"\$(\w+)"
        matches = re.findall(pattern, expression)
        return list(set(matches))  # 去重

    def get_complexity_score(self, expression: str) -> float:
        """
        计算表达式的复杂度分数

        复杂度基于：
        1. 表达式长度
        2. 函数调用数量
        3. 嵌套层数

        Args:
            expression: 因子表达式

        Returns:
            float: 复杂度分数（0-100）

        Examples:
            >>> parser = FactorExpressionParser()
            >>> parser.get_complexity_score("$close")
            1.0
            >>> parser.get_complexity_score("Ref($close, 5) / Ref($close, 20) - 1")
            15.0
        """
        # 基础分数：表达式长度（每字符0.1分）
        length_score = len(expression) * 0.1

        # 函数调用分数（每个函数10分）
        function_pattern = r"\w+\s*\("
        function_count = len(re.findall(function_pattern, expression))
        function_score = function_count * 10

        # 嵌套层数（每层5分）
        max_nesting = 0
        current_nesting = 0
        for char in expression:
            if char == "(":
                current_nesting += 1
                max_nesting = max(max_nesting, current_nesting)
            elif char == ")":
                current_nesting -= 1

        nesting_score = max_nesting * 5

        total_score = length_score + function_score + nesting_score
        return min(total_score, 100.0)  # 最大100分


# 便捷函数
def validate_factor_expression(expression: str, custom_patterns: Optional[List[str]] = None) -> bool:
    """
    验证因子表达式是否包含未来函数

    这是一个便捷函数，直接调用 FactorExpressionParser.validate_no_future_functions

    Args:
        expression: 因子表达式字符串
        custom_patterns: 自定义的未来函数模式列表

    Returns:
        bool: 如果表达式安全返回 True

    Raises:
        ValueError: 如果检测到未来函数

    Examples:
        >>> validate_factor_expression("Ref($close, 5)")
        True
        >>> try:
        ...     validate_factor_expression("Ref($close, -5)")
        ... except ValueError:
        ...     print("检测到未来函数")
        检测到未来函数
    """
    parser = FactorExpressionParser(custom_patterns)
    return parser.validate_no_future_functions(expression)


if __name__ == "__main__":
    # 示例：使用因子表达式解析器
    parser = FactorExpressionParser()

    print("=" * 60)
    print("因子表达式解析器示例")
    print("=" * 60)

    # 示例1：安全的表达式
    print("\n示例1：安全的表达式")
    safe_expr = "Ref($close, 20) / Ref($close, 5) - 1"
    print(f"表达式: {safe_expr}")
    try:
        if parser.validate_no_future_functions(safe_expr):
            print("✓ 表达式安全")
    except ValueError as e:
        print(f"✗ 错误: {e}")

    # 示例2：危险的表达式
    print("\n示例2：危险的表达式（包含未来函数）")
    dangerous_expr = "Ref($close, -2) / Ref($close, -1) - 1"
    print(f"表达式: {dangerous_expr}")
    try:
        if parser.validate_no_future_functions(dangerous_expr):
            print("✓ 表达式安全")
    except ValueError as e:
        print(f"✗ 检测到错误:\n{e}")

    # 示例3：提取字段
    print("\n示例3：提取表达式中的字段")
    expr = "($close - Ref($low, 5)) / Ref($volume, 10)"
    fields = parser.extract_fields(expr)
    print(f"表达式: {expr}")
    print(f"使用的字段: {fields}")

    # 示例4：复杂度分析
    print("\n示例4：复杂度分析")
    simple_expr = "$close"
    complex_expr = "Ref($close, 5) / Ref($close, 20) - 1 + Mean($volume, 10)"
    print(f"简单表达式: {simple_expr}")
    print(f"复杂度分数: {parser.get_complexity_score(simple_expr):.1f}")
    print(f"\n复杂表达式: {complex_expr}")
    print(f"复杂度分数: {parser.get_complexity_score(complex_expr):.1f}")

    # 示例5：安全性检查
    print("\n示例5：安全性检查（返回警告而非异常）")
    expr = "Ref($close, 5)"
    is_safe, warnings = parser.check_expression_safety(expr)
    print(f"表达式: {expr}")
    print(f"是否安全: {is_safe}")
    print(f"警告: {warnings if warnings else '无'}")
