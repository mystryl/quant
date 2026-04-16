"""
未来函数保护器单元测试

测试因子表达式解析器的未来函数检测功能。
"""

import pytest
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.guard import (
    FactorExpressionParser,
    FutureFunctionError,
    validate_expression,
    analyze_expression
)


class TestFactorExpressionParser:
    """测试因子表达式解析器"""

    def setup_method(self):
        """设置测试环境"""
        self.parser = FactorExpressionParser(strict_mode=False)

    def test_safe_expressions(self):
        """测试安全的表达式"""
        safe_expressions = [
            "Ref($close, 1)",
            "Ref($close, 5)",
            "Mean($close, 20)",
            "Std($close, 30) / Mean($close, 30)",
            "$close[0]",
            "$close[1]",
            "$close[5]",
            "Roll($close, -1)",
            "Mean($close, 20) / Std($close, 30)",
            "($close - $open) / $open",
        ]

        for expr in safe_expressions:
            assert self.parser.validate_no_future_functions(expr) is True, \
                f"表达式应该是安全的: {expr}"

    def test_unsafe_expressions_with_ref_negative(self):
        """测试 Ref() 使用负数"""
        unsafe_expressions = [
            "Ref($close, -1)",
            "Ref($close, -5)",
            "Ref($open, -2)",
            "Ref($volume, -10)",
        ]

        for expr in unsafe_expressions:
            with pytest.raises(FutureFunctionError) as exc_info:
                self.parser.validate_no_future_functions(expr)
            assert expr in str(exc_info.value)

    def test_unsafe_expressions_with_negative_index(self):
        """测试负索引"""
        unsafe_expressions = [
            "$close[-1]",
            "$close[-5]",
            "$open[-2]",
            "$volume[-10]",
        ]

        for expr in unsafe_expressions:
            with pytest.raises(FutureFunctionError) as exc_info:
                self.parser.validate_no_future_functions(expr)
            assert expr in str(exc_info.value)

    def test_unsafe_expressions_with_roll_positive(self):
        """测试 Roll() 使用正数"""
        unsafe_expressions = [
            "Roll($close, 1)",
            "Roll($close, 5)",
            "Roll($open, 2)",
        ]

        for expr in unsafe_expressions:
            with pytest.raises(FutureFunctionError) as exc_info:
                self.parser.validate_no_future_functions(expr)
            assert expr in str(exc_info.value)

    def test_unsafe_expressions_with_shift_negative(self):
        """测试 Shift() 使用负数"""
        unsafe_expressions = [
            "Shift($close, -1)",
            "Shift($close, -5)",
        ]

        for expr in unsafe_expressions:
            with pytest.raises(FutureFunctionError) as exc_info:
                self.parser.validate_no_future_functions(expr)
            assert expr in str(exc_info.value)

    def test_complex_safe_expressions(self):
        """测试复杂的安全表达式"""
        safe_expressions = [
            # 动量因子
            "Ref($close, 1) / Ref($close, 5) - 1",
            # 波动率因子
            "Std($close, 20) / Mean($close, 20)",
            # 换手率因子
            "$volume / Mean($volume, 20)",
            # 价格动量
            "($close - Ref($close, 20)) / Ref($close, 20)",
            # 复合因子
            "(Ref($close, 1) - Ref($open, 1)) / Ref($open, 1)",
        ]

        for expr in safe_expressions:
            assert self.parser.validate_no_future_functions(expr) is True, \
                f"复杂表达式应该是安全的: {expr}"

    def test_complex_unsafe_expressions(self):
        """测试复杂的不安全表达式"""
        unsafe_expressions = [
            # 包含未来引用的动量因子
            "Ref($close, -1) / Ref($close, -5) - 1",
            # 混合使用
            "Ref($close, 1) + Ref($open, -1)",
            # 多个未来引用
            "($close[-1] - Ref($close, -2)) / $open",
        ]

        for expr in unsafe_expressions:
            with pytest.raises(FutureFunctionError):
                self.parser.validate_no_future_functions(expr)

    def test_extract_variables(self):
        """测试变量提取"""
        test_cases = [
            ("Ref($close, 1)", ["$close"]),
            ("Ref($close, 1) + $volume", ["$close", "$volume"]),
            ("Mean($close, 20) / Std($close, 30)", ["$close"]),
            ("($close - $open) / $volume", ["$close", "$open", "$volume"]),
            ("$close[1]", ["$close"]),
        ]

        for expr, expected_vars in test_cases:
            vars = self.parser.extract_variables(expr)
            assert vars == expected_vars, \
                f"表达式 {expr} 的变量应该是 {expected_vars}, 实际是 {vars}"

    def test_analyze_expression_safe(self):
        """测试分析安全表达式"""
        expr = "Mean($close, 20) / Std($close, 30)"
        result = self.parser.analyze_expression(expr)

        assert result['safe'] is True
        assert len(result['future_functions']) == 0
        assert len(result['recommendations']) == 0

    def test_analyze_expression_unsafe(self):
        """测试分析不安全表达式"""
        expr = "Ref($close, -1)"
        result = self.parser.analyze_expression(expr)

        assert result['safe'] is False
        assert len(result['future_functions']) == 1
        assert len(result['recommendations']) > 0

    def test_get_expression_info(self):
        """测试获取表达式完整信息"""
        expr = "Ref($close, 1) + $volume"
        info = self.parser.get_expression_info(expr)

        assert info['expression'] == expr
        assert info['safe'] is True
        assert set(info['variables']) == {'$close', '$volume'}
        assert info['future_function_count'] == 0
        assert info['has_future_functions'] is False

    def test_error_message_format(self):
        """测试错误消息格式"""
        expr = "Ref($close, -1)"

        try:
            self.parser.validate_no_future_functions(expr)
            assert False, "应该抛出 FutureFunctionError"
        except FutureFunctionError as e:
            error_msg = str(e)
            # 检查错误消息包含关键信息
            assert "未来函数" in error_msg
            assert expr in error_msg
            assert "Ref" in error_msg
            assert "建议" in error_msg

    def test_strict_mode_suspicious_patterns(self):
        """测试严格模式下的可疑模式检测"""
        parser_strict = FactorExpressionParser(strict_mode=True)

        # 包含可疑但可能合法的表达式
        expr = "Mean($close, -1)"  # 这里的 -1 是周期参数，不是未来引用

        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = parser_strict.analyze_expression(expr)

            # 严格模式下应该检测到可疑模式
            assert len(result['suspicious_patterns']) > 0 or len(w) > 0

    def test_empty_expression(self):
        """测试空表达式"""
        expr = ""
        assert self.parser.validate_no_future_functions(expr) is True

    def test_expression_without_variables(self):
        """测试没有变量的表达式"""
        expr = "1 + 2"
        assert self.parser.validate_no_future_functions(expr) is True


class TestConvenienceFunctions:
    """测试便捷函数"""

    def test_validate_expression(self):
        """测试 validate_expression 函数"""
        assert validate_expression("Ref($close, 1)") is True

        with pytest.raises(FutureFunctionError):
            validate_expression("Ref($close, -1)")

    def test_analyze_expression_function(self):
        """测试 analyze_expression 函数"""
        result = analyze_expression("Ref($close, 1)")
        assert result['safe'] is True

        result = analyze_expression("Ref($close, -1)")
        assert result['safe'] is False


class TestRealWorldScenarios:
    """测试真实场景"""

    def setup_method(self):
        """设置测试环境"""
        self.parser = FactorExpressionParser(strict_mode=False)

    def test_common_momentum_factor(self):
        """测试常见的动量因子"""
        # 动量因子：过去 N 天的收益率
        factors = [
            "Ref($close, 1) / Ref($close, 5) - 1",  # 5日动量
            "Ref($close, 1) / Ref($close, 20) - 1",  # 20日动量
            "Ref($close, 1) / Ref($close, 60) - 1",  # 60日动量
        ]

        for factor in factors:
            assert self.parser.validate_no_future_functions(factor) is True, \
                f"动量因子应该是安全的: {factor}"

    def test_common_volatility_factor(self):
        """测试常见的波动率因子"""
        # 波动率因子：过去 N 天的标准差
        factors = [
            "Std($close, 20) / Mean($close, 20)",
            "Std($close, 30) / Mean($close, 30)",
            "Std($close, 60) / Mean($close, 60)",
        ]

        for factor in factors:
            assert self.parser.validate_no_future_functions(factor) is True, \
                f"波动率因子应该是安全的: {factor}"

    def test_common_reversal_factor(self):
        """测试常见的反转因子"""
        # 反转因子：过去一段时间的表现
        factors = [
            "Ref($close, 1) - Ref($close, 5)",
            "Ref($close, 1) - Ref($close, 10)",
            "($close - Ref($close, 5)) / Ref($close, 5)",
        ]

        for factor in factors:
            assert self.parser.validate_no_future_functions(factor) is True, \
                f"反转因子应该是安全的: {factor}"

    def test_volume_factors(self):
        """测试成交量因子"""
        factors = [
            "$volume / Mean($volume, 20)",
            "Std($volume, 20) / Mean($volume, 20)",
            "$volume / Ref($volume, 1)",
        ]

        for factor in factors:
            assert self.parser.validate_no_future_functions(factor) is True, \
                f"成交量因子应该是安全的: {factor}"

    def test_incorrect_future_lookahead(self):
        """测试错误的未来引用"""
        # 这些是常见的错误，应该被检测出来
        incorrect_factors = [
            "Ref($close, -1) / Ref($close, -5) - 1",  # 错误的动量因子
            "$close[-1] / $close[-5] - 1",  # 错误的索引
            "Ref($close, -10)",  # 直接使用未来数据
            "Roll($close, 5)",  # Roll 使用正数
        ]

        for factor in incorrect_factors:
            with pytest.raises(FutureFunctionError):
                self.parser.validate_no_future_functions(factor)


def run_tests():
    """运行所有测试"""
    print("=" * 80)
    print("运行未来函数保护器单元测试")
    print("=" * 80)

    # 运行 pytest
    pytest.main([__file__, "-v", "--tb=short"])


if __name__ == "__main__":
    run_tests()
