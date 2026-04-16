"""
未来函数保护器 - 因子表达式验证模块

该模块提供了静态分析工具，用于检测因子表达式中的未来函数（look-ahead bias）。
未来函数是指在计算因子时使用了未来数据，这会导致回测结果过于乐观，实际交易时无法实现。

核心功能：
1. 检测 Ref() 函数的未来引用
2. 检测负索引的未来引用
3. 检测 Roll() 函数的未来偏移
4. 提供清晰的错误提示

设计原则：
- 宁可误报，不可漏报
- 提供详细的错误信息和建议
- 支持常见的因子表达式格式
"""

import re
from typing import List, Tuple
from dataclasses import dataclass


@dataclass
class FutureFunctionMatch:
    """未来函数匹配结果"""

    pattern: str
    match_text: str
    position: Tuple[int, int]
    suggestion: str


class FutureFunctionError(Exception):
    """未来函数检测异常"""

    def __init__(self, expression: str, matches: List[FutureFunctionMatch]):
        self.expression = expression
        self.matches = matches
        super().__init__(self._format_error_message())

    def _format_error_message(self) -> str:
        """格式化错误消息"""
        msg = [
            "\n" + "=" * 80,
            "❌ 因子表达式包含未来函数（Look-ahead Bias）",
            "=" * 80,
            f"\n表达式: {self.expression}",
            f"\n检测到 {len(self.matches)} 个未来函数引用:\n",
        ]

        for i, match in enumerate(self.matches, 1):
            msg.append(f"  {i}. 模式: {match.pattern}")
            msg.append(f"     位置: {match.match_text}")
            msg.append(f"     建议: {match.suggestion}\n")

        msg.append("原因说明:")
        msg.append("  因子计算时只能使用历史数据，不能引用未来数据。")
        msg.append("  使用未来数据会导致回测结果过于乐观，实际交易时无法实现。\n")

        msg.append("修复建议:")
        msg.append("  1. 检查 Ref() 的第二个参数，确保使用正数（引用历史）")
        msg.append("  2. 检查数组索引，确保使用非负整数")
        msg.append("  3. 检查 Roll() 的偏移量，确保不引用未来数据\n")

        msg.append("=" * 80 + "\n")

        return "\n".join(msg)


class FactorExpressionParser:
    """
    因子表达式解析器 - 检测未来函数

    该类使用静态分析方法，通过正则表达式检测因子表达式中的未来函数模式。

    支持的检测模式：
    1. Ref($close, -N) - 使用负数引用未来数据
    2. $close[-N] - 使用负索引引用未来数据
    3. Roll($close, N) - 使用正偏移引用未来数据
    4. 其他常见的未来数据引用模式

    示例:
        >>> parser = FactorExpressionParser()
        >>>
        >>> # 检测未来函数 - 会抛出异常
        >>> parser.validate_no_future_functions("Ref($close, -1)")
        >>>
        >>> # 正确的表达式 - 不会抛出异常
        >>> parser.validate_no_future_functions("Ref($close, 1)")
    """

    # 未来函数模式定义
    # 格式: (正则表达式, 描述, 建议)
    FUTURE_PATTERNS = [
        (
            r"Ref\s*\(\s*\$?\w+\s*,\s*-\s*\d+\s*\)",
            "Ref() 使用负数参数",
            "Ref 的第二个参数应该为正数，例如 Ref($close, 1) 引用前一日收盘价",
        ),
        (r"\$\w+\[\s*-\s*\d+\s*\]", "数组使用负索引", "数组索引应该使用非负整数，例如 $close[1] 表示前一日收盘价"),
        (
            r"Roll\s*\(\s*\$?\w+\s*,\s*[1-9]\d*\s*\)",
            "Roll() 使用正偏移",
            "Roll 的第二个参数应该为 0 或负数，例如 Roll($close, -1) 向后滚动",
        ),
        (
            r"Shift\s*\(\s*\$?\w+\s*,\s*-\s*\d+\s*\)",
            "Shift() 使用负数",
            "Shift 的第二个参数应该为正数，例如 Shift($close, 1) 向后移动",
        ),
    ]

    # 需要特殊关注的可疑模式（可能是未来函数）
    SUSPICIOUS_PATTERNS = [
        (r"\[\s*-\s*\d+\s*\]", "负索引", "检查是否在引用未来数据"),
        (r",\s*-\s*\d+\s*\)", "负数参数", "检查该参数是否表示未来引用"),
    ]

    def __init__(self, strict_mode: bool = True):
        """
        初始化解析器

        Args:
            strict_mode: 严格模式，如果为 True，检测到任何可疑模式都会警告
        """
        self.strict_mode = strict_mode

    def validate_no_future_functions(self, expression: str) -> bool:
        """
        验证因子表达式不包含未来函数

        该方法会检查表达式中的所有未来函数模式，如果检测到任何未来函数，
        会抛出 FutureFunctionError 异常。

        Args:
            expression: 因子表达式字符串
                例如: "Ref($close, 1) / Ref($open, 1)"

        Returns:
            bool: 如果没有检测到未来函数，返回 True

        Raises:
            FutureFunctionError: 当检测到未来函数时

        Examples:
            >>> parser = FactorExpressionParser()
            >>> parser.validate_no_future_functions("Ref($close, 1)")
            True

            >>> try:
            ...     parser.validate_no_future_functions("Ref($close, -1)")
            ... except FutureFunctionError as e:
            ...     print(e)
            检测到未来函数...
        """
        matches = []

        # 检查所有未来函数模式
        for pattern, description, suggestion in self.FUTURE_PATTERNS:
            for match in re.finditer(pattern, expression):
                matches.append(
                    FutureFunctionMatch(
                        pattern=description,
                        match_text=match.group(),
                        position=(match.start(), match.end()),
                        suggestion=suggestion,
                    )
                )

        # 如果检测到未来函数，抛出异常
        if matches:
            raise FutureFunctionError(expression, matches)

        # 在严格模式下，检查可疑模式
        if self.strict_mode:
            suspicious_matches = self._check_suspicious_patterns(expression)
            if suspicious_matches:
                self._warn_suspicious_patterns(expression, suspicious_matches)

        return True

    def _check_suspicious_patterns(self, expression: str) -> List[FutureFunctionMatch]:
        """检查可疑模式"""
        matches = []

        for pattern, description, suggestion in self.SUSPICIOUS_PATTERNS:
            for match in re.finditer(pattern, expression):
                matches.append(
                    FutureFunctionMatch(
                        pattern=description,
                        match_text=match.group(),
                        position=(match.start(), match.end()),
                        suggestion=suggestion,
                    )
                )

        return matches

    def _warn_suspicious_patterns(self, expression: str, matches: List[FutureFunctionMatch]):
        """警告可疑模式"""
        import warnings

        msg = [
            "\n" + "=" * 80,
            "⚠️  检测到可疑的未来函数模式",
            "=" * 80,
            f"\n表达式: {expression}",
            f"\n发现 {len(matches)} 个可疑模式:\n",
        ]

        for i, match in enumerate(matches, 1):
            msg.append(f"  {i}. 模式: {match.pattern}")
            msg.append(f"     位置: {match.match_text}")
            msg.append(f"     建议: {match.suggestion}\n")

        msg.append("这些模式可能是正常的，请仔细检查。")
        msg.append("如果确认没有问题，可以设置 strict_mode=False 来禁用此警告。")
        msg.append("=" * 80 + "\n")

        warnings.warn("\n".join(msg), UserWarning)

    def analyze_expression(self, expression: str) -> dict:
        """
        分析因子表达式，返回详细的分析结果

        Args:
            expression: 因子表达式字符串

        Returns:
            dict: 包含分析结果的字典
                - safe: bool, 是否安全（无未来函数）
                - future_functions: list, 检测到的未来函数
                - suspicious_patterns: list, 检测到的可疑模式
                - recommendations: list, 修复建议

        Examples:
            >>> parser = FactorExpressionParser()
            >>> result = parser.analyze_expression("Ref($close, 1)")
            >>> print(result['safe'])
            True
        """
        result = {"safe": True, "future_functions": [], "suspicious_patterns": [], "recommendations": []}

        # 检查未来函数
        for pattern, description, suggestion in self.FUTURE_PATTERNS:
            for match in re.finditer(pattern, expression):
                result["future_functions"].append(
                    {
                        "pattern": description,
                        "match": match.group(),
                        "position": (match.start(), match.end()),
                        "suggestion": suggestion,
                    }
                )

        # 检查可疑模式
        if self.strict_mode:
            for pattern, description, suggestion in self.SUSPICIOUS_PATTERNS:
                for match in re.finditer(pattern, expression):
                    result["suspicious_patterns"].append(
                        {
                            "pattern": description,
                            "match": match.group(),
                            "position": (match.start(), match.end()),
                            "suggestion": suggestion,
                        }
                    )

        # 判断是否安全
        result["safe"] = len(result["future_functions"]) == 0

        # 生成建议
        if result["future_functions"]:
            result["recommendations"].append("表达式包含未来函数，请修复以下问题：")
            for func in result["future_functions"]:
                result["recommendations"].append(f"  - {func['pattern']}: {func['suggestion']}")

        if result["suspicious_patterns"]:
            result["recommendations"].append("表达式包含可疑模式，请确认以下内容：")
            for pattern in result["suspicious_patterns"]:
                result["recommendations"].append(f"  - {pattern['pattern']}: {pattern['suggestion']}")

        return result

    def extract_variables(self, expression: str) -> List[str]:
        """
        提取表达式中的变量

        Args:
            expression: 因子表达式字符串

        Returns:
            list: 变量列表，例如 ['$close', '$open', '$volume']

        Examples:
            >>> parser = FactorExpressionParser()
            >>> vars = parser.extract_variables("Ref($close, 1) + $volume")
            >>> print(vars)
            ['$close', '$volume']
        """
        # 匹配 $variable 形式的变量
        pattern = r"\$\w+"
        variables = re.findall(pattern, expression)

        # 去重并保持顺序
        seen = set()
        unique_vars = []
        for var in variables:
            if var not in seen:
                seen.add(var)
                unique_vars.append(var)

        return unique_vars

    def get_expression_info(self, expression: str) -> dict:
        """
        获取表达式的完整信息

        Args:
            expression: 因子表达式字符串

        Returns:
            dict: 包含表达式所有信息的字典
        """
        analysis = self.analyze_expression(expression)

        return {
            "expression": expression,
            "variables": self.extract_variables(expression),
            "safe": analysis["safe"],
            "has_future_functions": len(analysis["future_functions"]) > 0,
            "has_suspicious_patterns": len(analysis["suspicious_patterns"]) > 0,
            "future_function_count": len(analysis["future_functions"]),
            "suspicious_pattern_count": len(analysis["suspicious_patterns"]),
            "recommendations": analysis["recommendations"],
        }


# 便捷函数
def validate_expression(expression: str, strict_mode: bool = True) -> bool:
    """
    验证因子表达式（便捷函数）

    Args:
        expression: 因子表达式字符串
        strict_mode: 是否使用严格模式

    Returns:
        bool: 如果表达式安全，返回 True

    Raises:
        FutureFunctionError: 当检测到未来函数时

    Examples:
        >>> validate_expression("Ref($close, 1)")
        True
    """
    parser = FactorExpressionParser(strict_mode=strict_mode)
    return parser.validate_no_future_functions(expression)


def analyze_expression(expression: str, strict_mode: bool = True) -> dict:
    """
    分析因子表达式（便捷函数）

    Args:
        expression: 因子表达式字符串
        strict_mode: 是否使用严格模式

    Returns:
        dict: 分析结果

    Examples:
        >>> result = analyze_expression("Ref($close, 1)")
        >>> print(result['safe'])
        True
    """
    parser = FactorExpressionParser(strict_mode=strict_mode)
    return parser.analyze_expression(expression)


# 测试用例
if __name__ == "__main__":
    # 创建测试用例
    test_cases = [
        # 安全的表达式
        ("Ref($close, 1)", True, "正确：引用前一日收盘价"),
        ("Ref($close, 5)", True, "正确：引用前5日收盘价"),
        ("$close[0]", True, "正确：引用当前收盘价"),
        ("$close[1]", True, "正确：引用前一日收盘价"),
        ("Roll($close, -1)", True, "正确：向后滚动"),
        ("Mean($close, 20)", True, "正确：计算20日均值"),
        ("Std($close, 30) / Mean($close, 30)", True, "正确：变异率因子"),
        # 不安全的表达式（未来函数）
        ("Ref($close, -1)", False, "错误：引用未来一日收盘价"),
        ("Ref($close, -5)", False, "错误：引用未来5日收盘价"),
        ("$close[-1]", False, "错误：使用负索引引用未来数据"),
        ("$close[-5]", False, "错误：使用负索引引用未来数据"),
        ("Roll($close, 1)", False, "错误：向前滚动引用未来数据"),
    ]

    parser = FactorExpressionParser(strict_mode=False)

    print("=" * 80)
    print("因子表达式未来函数检测测试")
    print("=" * 80)

    for expr, should_be_safe, description in test_cases:
        print(f"\n测试: {description}")
        print(f"表达式: {expr}")

        try:
            parser.validate_no_future_functions(expr)
            if should_be_safe:
                print("✅ 通过：表达式安全")
            else:
                print("❌ 失败：应该检测到未来函数但没有")
        except FutureFunctionError as e:
            if not should_be_safe:
                print("✅ 通过：成功检测到未来函数")
            else:
                print(f"❌ 失败：误报 - {e}")

    print("\n" + "=" * 80)
    print("表达式分析示例")
    print("=" * 80)

    # 安全的表达式分析
    safe_expr = "Mean($close, 20) / Std($close, 20)"
    print(f"\n表达式: {safe_expr}")
    info = parser.get_expression_info(safe_expr)
    print(f"安全: {info['safe']}")
    print(f"变量: {info['variables']}")
    print(f"未来函数数量: {info['future_function_count']}")

    # 不安全的表达式分析
    unsafe_expr = "Ref($close, -1) + Ref($open, -2)"
    print(f"\n表达式: {unsafe_expr}")
    info = parser.get_expression_info(unsafe_expr)
    print(f"安全: {info['safe']}")
    print(f"变量: {info['variables']}")
    print(f"未来函数数量: {info['future_function_count']}")
    if info["recommendations"]:
        print("建议:")
        for rec in info["recommendations"]:
            print(f"  {rec}")
