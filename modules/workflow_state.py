"""
工作流状态管理器
管理论文到Beamer转换过程中的所有中间产物，确保Agent间数据共享
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict

@dataclass
class WorkflowState:
    """工作流状态数据类，管理所有中间产物路径和元数据"""
    
    # 基础信息
    session_id: str
    original_pdf_path: str
    output_base_dir: str
    
    # 各阶段输出路径
    parser_output_path: Optional[str] = None
    planner_output_path: Optional[str] = None
    tex_output_path: Optional[str] = None
    pdf_output_path: Optional[str] = None
    verification_report_path: Optional[str] = None
    speech_output_path: Optional[str] = None
    
    # 目录路径
    raw_dir: Optional[str] = None
    plan_dir: Optional[str] = None
    tex_dir: Optional[str] = None
    images_dir: Optional[str] = None
    verification_dir: Optional[str] = None
    
    # 工作流元数据
    language: str = "zh"
    model_name: str = "gpt-4o"
    theme: str = "Madrid"
    
    # 状态标记
    parser_completed: bool = False
    planner_completed: bool = False
    verification_completed: bool = False
    tex_completed: bool = False
    speech_completed: bool = False
    
    def __post_init__(self):
        """初始化后设置目录结构"""
        if not self.raw_dir:
            self.raw_dir = os.path.join(self.output_base_dir, "raw", self.session_id)
        if not self.plan_dir:
            self.plan_dir = os.path.join(self.output_base_dir, "plan", self.session_id)
        if not self.tex_dir:
            self.tex_dir = os.path.join(self.output_base_dir, "tex", self.session_id)
        if not self.images_dir:
            self.images_dir = os.path.join(self.output_base_dir, "images", self.session_id)
        if not self.verification_dir:
            self.verification_dir = os.path.join(self.output_base_dir, "verification", self.session_id)
            
        # 创建目录
        self._ensure_directories()
    
    def _ensure_directories(self):
        """确保所有必要目录存在"""
        for dir_path in [self.raw_dir, self.plan_dir, self.tex_dir, self.images_dir, self.verification_dir]:
            os.makedirs(dir_path, exist_ok=True)
    
    def set_parser_output(self, output_path: str):
        """设置Parser输出路径并标记完成"""
        self.parser_output_path = output_path
        self.parser_completed = True
    
    def set_planner_output(self, output_path: str):
        """设置Planner输出路径并标记完成"""
        self.planner_output_path = output_path
        self.planner_completed = True
    
    def set_tex_output(self, tex_path: str, pdf_path: str = None):
        """设置TEX输出路径并标记完成"""
        self.tex_output_path = tex_path
        if pdf_path:
            self.pdf_output_path = pdf_path
        self.tex_completed = True
    
    def set_verification_output(self, verification_report_path: str, verification_passed: bool):
        """设置验证阶段的输出"""
        self.verification_report_path = verification_report_path
        self.verification_passed = verification_passed
        self.verification_completed = True
        self._save_state()
    
    def set_repair_output(self, repair_report_path: str, repaired_plan_path: str, repair_success: bool):
        """设置修复阶段的输出"""
        self.repair_report_path = repair_report_path
        self.repaired_plan_path = repaired_plan_path
        self.repair_success = repair_success
        self.repair_completed = True
        self._save_state()
    
    def set_speech_output(self, speech_path: str, speech_success: bool):
        """设置演讲稿生成阶段的输出"""
        self.speech_output_path = speech_path
        self.speech_success = speech_success
        self.speech_completed = True
        self._save_state()
    
    def _save_state(self):
        """保存工作流状态到文件"""
        try:
            state_file = os.path.join(self.output_base_dir, f"workflow_state_{self.session_id}.json")
            with open(state_file, 'w', encoding='utf-8') as f:
                json.dump(asdict(self), f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save workflow state: {e}")
    
    def get_parser_content(self) -> Optional[Dict[str, Any]]:
        """获取Parser解析的内容"""
        if not self.parser_output_path or not os.path.exists(self.parser_output_path):
            return None
        
        try:
            with open(self.parser_output_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"Failed to load parser content: {e}")
            return None
    
    def get_planner_content(self) -> Optional[Dict[str, Any]]:
        """获取Planner生成的计划"""
        if not self.planner_output_path or not os.path.exists(self.planner_output_path):
            return None
        
        try:
            with open(self.planner_output_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"Failed to load planner content: {e}")
            return None
    
    def get_verification_report(self) -> Optional[Dict[str, Any]]:
        """获取验证报告"""
        if not self.verification_report_path or not os.path.exists(self.verification_report_path):
            return None
        
        try:
            with open(self.verification_report_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"Failed to load verification report: {e}")
            return None
    
    def save_state(self, state_file: str = None) -> str:
        """保存工作流状态到文件"""
        if not state_file:
            state_file = os.path.join(self.output_base_dir, f"workflow_state_{self.session_id}.json")
        
        try:
            with open(state_file, 'w', encoding='utf-8') as f:
                json.dump(asdict(self), f, ensure_ascii=False, indent=2)
            return state_file
        except Exception as e:
            logging.error(f"Failed to save workflow state: {e}")
            return ""
    
    @classmethod
    def load_state(cls, state_file: str) -> Optional['WorkflowState']:
        """从文件加载工作流状态"""
        if not os.path.exists(state_file):
            return None
        
        try:
            with open(state_file, 'r', encoding='utf-8') as f:
                state_data = json.load(f)
            return cls(**state_data)
        except Exception as e:
            logging.error(f"Failed to load workflow state: {e}")
            return None
    
    def is_ready_for_reference_search(self) -> bool:
        """检查是否准备好进行引用检索"""
        return self.parser_completed and self.parser_output_path is not None
    
    def get_reference_search_context(self, target_concept: str) -> Dict[str, Any]:
        """为引用检索准备上下文信息"""
        parser_content = self.get_parser_content()
        planner_content = self.get_planner_content()
        
        context = {
            "original_paper_path": self.parser_output_path,
            "target_concept": target_concept,
            "session_id": self.session_id,
            "output_dir": os.path.join(self.output_base_dir, "reference_enhancement", self.session_id)
        }
        
        # 确保引用检索输出目录存在
        os.makedirs(context["output_dir"], exist_ok=True)
        
        # 添加演示计划上下文（如果可用）
        if planner_content:
            context["presentation_context"] = planner_content
        
        return context
    
    def __str__(self) -> str:
        """字符串表示"""
        return f"WorkflowState(session_id={self.session_id}, parser={self.parser_completed}, planner={self.planner_completed}, tex={self.tex_completed})"


class WorkflowStateManager:
    """工作流状态管理器，提供全局状态管理功能"""
    
    def __init__(self):
        self.active_states: Dict[str, WorkflowState] = {}
        self.logger = logging.getLogger(__name__)
    
    def create_workflow(self, session_id: str, original_pdf_path: str, 
                       output_base_dir: str, **kwargs) -> WorkflowState:
        """创建新的工作流状态"""
        workflow_state = WorkflowState(
            session_id=session_id,
            original_pdf_path=original_pdf_path,
            output_base_dir=output_base_dir,
            **kwargs
        )
        
        self.active_states[session_id] = workflow_state
        self.logger.info(f"Created new workflow state: {session_id}")
        
        return workflow_state
    
    def get_workflow(self, session_id: str) -> Optional[WorkflowState]:
        """获取指定的工作流状态"""
        return self.active_states.get(session_id)
    
    def save_all_states(self, base_dir: str):
        """保存所有活跃的工作流状态"""
        for session_id, state in self.active_states.items():
            state_file = os.path.join(base_dir, f"workflow_state_{session_id}.json")
            state.save_state(state_file)
    
    def load_workflow_from_file(self, state_file: str) -> Optional[WorkflowState]:
        """从文件加载工作流状态"""
        state = WorkflowState.load_state(state_file)
        if state:
            self.active_states[state.session_id] = state
            self.logger.info(f"Loaded workflow state: {state.session_id}")
        return state
    
    def cleanup_workflow(self, session_id: str):
        """清理指定的工作流状态"""
        if session_id in self.active_states:
            del self.active_states[session_id]
            self.logger.info(f"Cleaned up workflow state: {session_id}")


# 全局状态管理器实例
workflow_manager = WorkflowStateManager()

