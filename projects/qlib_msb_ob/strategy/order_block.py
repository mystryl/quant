#!/usr/bin/env python3
"""
订单块(Order Block)管理模块

对应 Pine 源码中的 OrderBlock 类型和管理逻辑
"""
import pandas as pd
from datetime import datetime
from typing import List, Optional
import uuid


class OrderBlock:
    """
    订单块类

    对应 Pine 源码第79-89行：
        type OrderBlock
            box        boxId
            label      labelId
            line       pocLineId
            float      qualityScore
            float      top
            float      bottom
            bool       isBull
            bool       isHPZ = false
            bool       mitigated = false
            int        mitigationBar = 0
    """

    def __init__(
        self,
        ob_id: str,
        ob_type: str,  # 'bullish' or 'bearish'
        timestamp: pd.Timestamp,
        ob_index: int,
        top: float,
        bottom: float,
        quality_score: float,
        is_hpz: bool = False
    ):
        self.id = ob_id
        self.type = ob_type  # 'bullish' or 'bearish' (对应Pine的isBull)
        self.timestamp = timestamp
        self.ob_index = ob_index
        self.top = top
        self.bottom = bottom
        self.poc = (top + bottom) / 2  # Point of Control (第196行)
        self.width = top - bottom
        self.quality_score = quality_score  # 对应Pine的qualityScore
        self.is_hpz = is_hpz  # 对应Pine的isHPZ
        self.is_mitigated = False  # 对应Pine的mitigated
        self.mitigation_timestamp = None
        self.mitigation_bar_index = None  # 对应Pine的mitigationBar

    def __repr__(self):
        return (f"OrderBlock(id={self.id[:8]}, type={self.type}, "
                f"top={self.top:.2f}, bottom={self.bottom:.2f}, "
                f"score={self.quality_score:.1f}, hpz={self.is_hpz}, "
                f"mitigated={self.is_mitigated})")


class OBManager:
    """
    订单块管理器

    对应 Pine 源码第91行和第236-294行的管理逻辑
    """

    def __init__(self, max_ob_count: int = 10):
        """
        初始化OB管理器

        Args:
            max_ob_count: 最大活跃OB数量（对应Pine源码第53行obCountInput）
        """
        self.max_ob_count = max_ob_count
        self.active_obs: List[OrderBlock] = []
        self.mitigated_obs: List[OrderBlock] = []
        self.total_ob_created = 0
        self.total_ob_mitigated = 0

    def add_ob(self, ob: OrderBlock):
        """
        添加新的OB

        对应 Pine 源码第228-229行：
            OrderBlock newOB = OrderBlock.new(...)
            obArray.push(newOB)

        和第289-293行的数量限制：
            if obArray.size() > obCountInput
                OrderBlock oldOB = obArray.shift()
        """
        self.active_obs.append(ob)
        self.total_ob_created += 1

        # 检查数量限制（第289-293行）
        if len(self.active_obs) > self.max_ob_count:
            old_ob = self.active_obs.pop(0)  # 移除最旧的OB
            # 注意：Pine源码会删除box/label/line，这里不需要

    def check_mitigation(
        self,
        current_high: float,
        current_low: float,
        current_timestamp: pd.Timestamp,
        current_bar_index: int
    ) -> List[OrderBlock]:
        """
        检查OB是否失效

        对应 Pine 源码第252-266行：
            bool isMitigated = ob.isBull ? low < ob.bottom : high > ob.top
            if isMitigated
                ob.mitigated := true
                ob.mitigationBar := barindex
                totalMitigated += 1

        Args:
            current_high: 当前K线最高价
            current_low: 当前K线最低价
            current_timestamp: 当前时间戳
            current_bar_index: 当前K线索引

        Returns:
            本轮新失效的OB列表
        """
        newly_mitigated = []

        for ob in self.active_obs[:]:  # 复制列表以安全迭代
            if ob.is_mitigated:
                continue

            # Pine 源码第252行
            if ob.type == 'bullish':
                # 看涨OB失效：low < ob.bottom（价格跌破OB底部）
                is_mitigated = current_low < ob.bottom
            else:  # bearish
                # 看跌OB失效：high > ob.top（价格突破OB顶部）
                is_mitigated = current_high > ob.top

            if is_mitigated:
                ob.is_mitigated = True
                ob.mitigation_timestamp = current_timestamp
                ob.mitigation_bar_index = current_bar_index
                self.total_ob_mitigated += 1

                # 移动到失效列表
                self.active_obs.remove(ob)
                self.mitigated_obs.append(ob)
                newly_mitigated.append(ob)

        return newly_mitigated

    def get_active_obs(self) -> List[OrderBlock]:
        """获取所有活跃的OB"""
        return self.active_obs.copy()

    def get_hpz_count(self) -> int:
        """
        获取当前活跃的HPZ-OB数量

        对应 Pine 源码第308-312行：
            int hpzs = 0
            if obArray.size() > 0
                for i = 0 to obArray.size() - 1
                    if obArray.get(i).isHPZ and not obArray.get(i).mitigated
                        hpzs += 1
        """
        return sum(1 for ob in self.active_obs if ob.is_hpz and not ob.is_mitigated)

    def get_reliability(self) -> float:
        """
        获取OB可靠性

        对应 Pine 源码第307行：
            float efficiency = totalObs > 0 ? (totalMitigated / totalObs) * 100 : 0

        Returns:
            可靠性百分比 (0-100)
        """
        if self.total_ob_created == 0:
            return 0.0
        return (self.total_ob_mitigated / self.total_ob_created) * 100

    def get_ob_stats(self) -> dict:
        """
        获取OB统计信息

        Returns:
            包含以下键的字典：
            - total_ob_created: 总OB创建数
            - total_ob_mitigated: 总OB失效数
            - active_ob_count: 当前活跃OB数
            - hpz_ob_count: 当前活跃HPZ-OB数
            - ob_reliability: OB可靠性百分比
        """
        return {
            'total_ob_created': self.total_ob_created,
            'total_ob_mitigated': self.total_ob_mitigated,
            'active_ob_count': len(self.active_obs),
            'hpz_ob_count': self.get_hpz_count(),
            'ob_reliability': self.get_reliability()
        }
