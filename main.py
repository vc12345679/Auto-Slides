#!/usr/bin/env python3
"""
Paper to Beamer Conversion Tool - Main Program
"""

import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# Load patches
from patch_openai import patch_openai_client, patch_langchain_openai

# Load environment variables
from dotenv import load_dotenv
if os.path.exists(".env"):
    load_dotenv(".env")
elif os.path.exists("env.local"):
    load_dotenv("env.local")

# Apply patches
patch_openai_client()
patch_langchain_openai()

# Import modules
from modules.pdf_parser import extract_pdf_content
from modules.presentation_planner import generate_presentation_plan
from modules.tex_workflow import run_tex_workflow, run_revision_tex_workflow
from modules.workflow_state import WorkflowState

def setup_logging(verbose=False):
    """Set up logging level and format"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Convert academic paper PDF to Beamer presentation'
    )
    
    # Required arguments
    parser.add_argument(
        'pdf_path', 
        help='Input PDF file path'
    )
    
    # Optional arguments
    parser.add_argument(
        '--output-dir', '-o',
        default='output',
        help='Output directory'
    )
    parser.add_argument(
        '--language', '-l',
        choices=['zh', 'en'],
        default='en',
        help='Output language, zh for Chinese, en for English'
    )
    parser.add_argument(
        '--model', '-m',
        default='gpt-4o',
        help='Language model to use'
    )
    parser.add_argument(
        '--max-retries', '-r',
        type=int,
        default=5,
        help='Maximum retries when compilation fails'
    )
    parser.add_argument(
        '--skip-compilation', '-s',
        action='store_true',
        help='Skip PDF compilation (generate TEX only)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show verbose logs'
    )
    parser.add_argument(
        '--interactive', '-i',
        action='store_true',
        help='Enable interactive mode, allow users to optimize presentation plan through multi-turn dialogue'
    )
    # Add support for revision mode
    parser.add_argument(
        '--revise', '-R',
        action='store_true',
        help='Enable revision mode, allow users to provide feedback to modify generated presentations'
    )
    parser.add_argument(
        '--original-plan', 
        help='Original presentation plan JSON file path (used in revision mode)'
    )
    parser.add_argument(
        '--previous-tex', 
        help='Previous version TEX file path (used in revision mode)'
    )
    parser.add_argument(
        '--feedback', 
        help='User feedback content (used in revision mode)'
    )
    parser.add_argument(
        '--theme',
        default='Madrid',
        help='Beamer theme, such as Madrid, Berlin, Singapore, etc.'
    )
    parser.add_argument(
        '--disable-llm-enhancement',
        action='store_true',
        help='Disable LLM enhancement, use basic PDF parsing only'
    )
    parser.add_argument(
        '--no-interactive-revise',
        action='store_true',
        help='Disable ReAct mode interactive revision (enabled by default)'
    )
    parser.add_argument(
        '--enable-verification',
        action='store_true',
        default=True,
        help='Enable presentation plan verification agent (detect consistency and hallucination) [enabled by default]'
    )
    parser.add_argument(
        '--enable-auto-repair',
        action='store_true',
        default=True,
        help='Enable auto-repair agent (automatically fix issues based on verification results) [enabled by default]'
    )
    parser.add_argument(
        '--disable-verification',
        action='store_true',
        help='Disable verification and repair functions (fast mode)'
    )
    parser.add_argument(
        '--enable-speech',
        action='store_true',
        help='Enable speech generation agent (generate accompanying speech script)'
    )
    parser.add_argument(
        '--speech-duration',
        type=int,
        default=15,
        help='Target speech duration (minutes, default 15 minutes)'
    )
    parser.add_argument(
        '--speech-style',
        choices=['academic_conference', 'classroom', 'industry_presentation', 'public_talk'],
        default='academic_conference',
        help='Speech style type'
    )
    
    return parser.parse_args()

def interactive_dialog(planner, logger):
    """
    与用户进行交互式对话，优化演示计划
    
    Args:
        planner: 演示计划生成器实例
        logger: 日志记录器
        
    Returns:
        Dict: 优化后的演示计划
    """
    logger.info("Entering interactive mode. Enter feedback to improve plan. Type 'exit' to quit.")
    
    while True:
        user_input = input("\nEnter your feedback: ")
        
        # Check for exit
        if user_input.lower() in ['退出', 'exit', 'quit']:
            logger.info("Exiting interactive mode")
            break
            
        # Process user input
        logger.info("Processing feedback...")
        response, updated_plan = planner.continue_conversation(user_input)
        
        # Print model response
        print("\n==== Model Response ====")
        print(response)
        print("========================")
        
    return planner.presentation_plan

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 设置日志
    logger = setup_logging(args.verbose)
    
    # 检查API密钥
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.error("未设置OPENAI_API_KEY环境变量")
        return 1
    
    # 创建输出目录
    output_dir = args.output_dir
    
    # 使用唯一的会话ID来区分不同的运行
    session_id = f"{int(time.time())}"
    
    # 创建各阶段输出目录
    raw_dir = os.path.join(output_dir, "raw", session_id)
    plan_dir = os.path.join(output_dir, "plan", session_id)
    tex_dir = os.path.join(output_dir, "tex", session_id)
    img_dir = os.path.join(output_dir, "images", session_id)
    
    for dir_path in [raw_dir, plan_dir, tex_dir, img_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    # 创建工作流状态管理器
    workflow_state = WorkflowState(
        session_id=session_id,
        original_pdf_path=args.pdf_path,
        output_base_dir=output_dir
    )
    
    # 检查是否为修订模式
    if args.revise:
        # 验证修订模式的必要参数
        if not args.original_plan or not args.previous_tex or not args.feedback:
            logger.error("修订模式需要提供--original-plan, --previous-tex和--feedback参数")
            return 1
            
        # 检查文件是否存在
        if not os.path.exists(args.original_plan):
            logger.error(f"原始演示计划文件不存在: {args.original_plan}")
            return 1
            
        if not os.path.exists(args.previous_tex):
            logger.error(f"先前版本的TEX文件不存在: {args.previous_tex}")
            return 1
            
        # 运行修订版TEX工作流
        logger.info("启动修订模式...")
        
        success, message, pdf_path = run_revision_tex_workflow(
            original_plan_path=args.original_plan,
            previous_tex_path=args.previous_tex,
            user_feedback=args.feedback,
            output_dir=tex_dir,
            model_name=args.model,
            language=args.language,
            theme=args.theme,
            max_retries=args.max_retries
        )
        
        if success:
            logger.info(f"修订版TEX生成和编译成功: {message}")
            logger.info(f"生成的PDF文件: {pdf_path}")
            return 0
        else:
            logger.error(f"修订版TEX生成和编译失败: {message}")
            return 1
    
    # 非修订模式的原有逻辑
    # 检查输入文件
    if not os.path.exists(args.pdf_path):
        logger.error(f"PDF文件不存在: {args.pdf_path}")
        return 1
        
    # 步骤1: 提取PDF内容
    logger.info("步骤1: 提取PDF内容...")
    try:
        # 决定是否启用LLM增强
        enable_llm_enhancement = not args.disable_llm_enhancement and bool(api_key)
        
        if not enable_llm_enhancement:
            if args.disable_llm_enhancement:
                logger.info("用户禁用了LLM增强功能")
            else:
                logger.warning("未设置API密钥，将禁用LLM增强功能")
        
        pdf_content, raw_content_path = extract_pdf_content(
            pdf_path=args.pdf_path, 
            output_dir=raw_dir,
            enable_llm_enhancement=enable_llm_enhancement,
            model_name=args.model,
            api_key=api_key
        )
        if not pdf_content:
            logger.error("PDF内容提取失败")
            return 1
            
        logger.info(f"PDF内容已保存到: {raw_content_path}")
        
        # 更新工作流状态
        workflow_state.set_parser_output(raw_content_path)
        workflow_state.images_dir = img_dir
        
        # 检查是否成功使用了LLM增强
        if pdf_content.get("enhanced_content"):
            logger.info("✅ LLM增强内容提取成功")
            enhanced = pdf_content["enhanced_content"]
            logger.info(f"提取到 {len(enhanced.get('tables', []))} 个表格")
            logger.info(f"提取到 {len(enhanced.get('presentation_sections', {}))} 个演讲章节")
        else:
            logger.info("使用基础PDF解析（未启用LLM增强）")
    except Exception as e:
        logger.error(f"PDF内容提取失败: {str(e)}")
        return 1
            
    # 步骤2: 生成演示计划
    logger.info("步骤2: 生成演示计划...")
    try:
        presentation_plan, plan_path, planner = generate_presentation_plan(
            raw_content_path=raw_content_path,
            output_dir=plan_dir,
            model_name=args.model,
            language=args.language
        )
            
        if not presentation_plan:
            logger.error("演示计划生成失败")
            return 1
            
        logger.info(f"演示计划已保存到: {plan_path}")
        
        # 更新工作流状态
        workflow_state.set_planner_output(plan_path)
            
        # 如果启用了交互式模式，进入对话
        if args.interactive and planner:
            logger.info("开始交互式优化...")
            presentation_plan = interactive_dialog(planner, logger)
            
            # 保存优化后的计划
            plan_path = planner.save_presentation_plan(presentation_plan)
            logger.info(f"优化后的演示计划已保存到: {plan_path}")
            
            # 更新工作流状态
            workflow_state.set_planner_output(plan_path)
    except Exception as e:
        logger.error(f"演示计划生成失败: {str(e)}")
        return 1
    
    # 步骤2.5: 验证演示计划（使用简化验证Agent）
    verification_passed = True
    verification_report = None
    verification_report_path = None
    if args.enable_verification and not args.disable_verification:
        verification_dir = os.path.join(output_dir, "verification", session_id)
        os.makedirs(verification_dir, exist_ok=True)
        
        try:
            # 导入简化验证Agent
            from modules.simplified_verification_agent import verify_content_coverage
            
            logger.info("步骤2.5: 验证内容覆盖度...")
            logger.info("正在检查核心内容是否充分覆盖...")
            
            verification_passed, verification_report, verification_report_path = verify_content_coverage(
                original_content_path=raw_content_path,
                presentation_plan_path=plan_path,
                output_dir=verification_dir,
                model_name=args.model,
                language=args.language
            )
            
            # 更新工作流状态
            workflow_state.set_verification_output(verification_report_path, verification_passed)
            
            if verification_passed:
                logger.info("✅ 内容覆盖度验证通过")
                if verification_report_path:
                    logger.info(f"验证报告已保存到: {verification_report_path}")
            else:
                logger.warning("⚠️ 内容覆盖度不足，建议进行修复")
                if verification_report_path:
                    logger.warning(f"验证报告已保存到: {verification_report_path}")
                
                # 显示缺失内容摘要
                if verification_report and "missing_content" in verification_report:
                    missing_content = verification_report["missing_content"]
                    if missing_content:
                        logger.warning("缺失的重要内容:")
                        for item in missing_content[:3]:  # 只显示前3个
                            logger.warning(f"  - {item.get('area', 'Unknown')}: {item.get('missing_content', '')[:100]}...")
                
                # 对于内容覆盖不足，询问用户是否继续
                if verification_report and verification_report.get("missing_content"):
                    user_choice = input("\n发现内容覆盖不足，是否启用自动修复？(y/n): ").strip().lower()
                    if user_choice != 'y':
                        logger.info("用户选择跳过修复，继续生成")
                        verification_passed = True  # 允许继续
            
        except Exception as e:
            logger.warning(f"验证步骤失败，继续执行: {str(e)}")
            # 验证失败不影响主流程继续执行
            verification_passed = True  # 设为True以避免阻塞流程
    
    # 步骤2.6: 自动修复（使用简化修复Agent）
    repaired_plan_path = plan_path  # 默认使用原始计划
    if args.enable_auto_repair and not args.disable_verification and args.enable_verification and verification_report and not verification_passed:
        repair_dir = os.path.join(output_dir, "repair", session_id)
        os.makedirs(repair_dir, exist_ok=True)
        
        try:
            # 导入简化修复Agent
            from modules.simplified_repair_agent import repair_content_coverage
            
            logger.info("步骤2.6: 补充缺失内容...")
            logger.info("正在基于验证结果补充重要内容...")
            
            repair_success, repair_report, repaired_plan_path = repair_content_coverage(
                presentation_plan_path=plan_path,
                verification_report_path=verification_report_path,
                original_content_path=raw_content_path,
                output_dir=repair_dir,
                model_name=args.model,
                language=args.language
            )
            
            if repair_success:
                logger.info("✅ 内容补充完成")
                logger.info(f"补充后的计划已保存到: {repaired_plan_path}")
                
                # 显示修复摘要
                if repair_report and "repair_summary" in repair_report:
                    summary = repair_report["repair_summary"]
                    total_repairs = summary.get('total_repairs', 0)
                    logger.info(f"补充内容数量: {total_repairs}")
                    if total_repairs > 0:
                        logger.info("内容覆盖度已得到改善")
                
                # 更新工作流状态使用修复后的计划
                workflow_state.set_planner_output(repaired_plan_path)
                plan_path = repaired_plan_path  # 更新变量用于后续TEX生成
            else:
                logger.info("⚠️ 未找到需要补充的内容，或补充失败")
                logger.info("将继续使用原始演示计划")
            
        except Exception as e:
            logger.warning(f"自动修复步骤失败，继续执行: {str(e)}")
            # 修复失败不影响主流程继续执行
        
    # 步骤3: 并行生成TEX和演讲稿
    logger.info("步骤3: 生成和编译TEX...")
    
    # 3.1: TEX生成和编译
    try:
        success, message, pdf_path = run_tex_workflow(
            presentation_plan_path=plan_path,
            output_dir=tex_dir,
            model_name=args.model,
            language=args.language,
            theme=args.theme,
            max_retries=args.max_retries,
            skip_compilation=args.skip_compilation  # 只跳过编译，不跳过TEX生成
        )
        
        if success:
            logger.info(f"TEX生成和编译成功: {message}")
            logger.info(f"生成的PDF文件: {pdf_path}")
            
            # 更新工作流状态
            tex_files = [f for f in os.listdir(tex_dir) if f.endswith(".tex") and not f.endswith("_revised.tex")]
            if tex_files:
                tex_file_path = os.path.join(tex_dir, tex_files[0])
                workflow_state.set_tex_output(tex_file_path, pdf_path)
        
        # 3.2: 演讲稿生成（可选，与TEX生成并行）
        speech_success = False
        speech_path = None
        if args.enable_speech:
            try:
                # 导入演讲稿生成Agent
                from modules.speech_generator import generate_speech_for_presentation
                
                logger.info("步骤3.2: 生成演讲稿...")
                
                speech_dir = os.path.join(output_dir, "speech", session_id)
                os.makedirs(speech_dir, exist_ok=True)
                
                speech_success, speech_result, speech_path = generate_speech_for_presentation(
                    presentation_plan_path=plan_path,
                    output_dir=speech_dir,
                    original_content_path=raw_content_path,
                    target_duration_minutes=args.speech_duration,
                    presentation_style=args.speech_style,
                    audience_level="expert",
                    model_name=args.model
                )
                
                if speech_success:
                    logger.info("✅ 演讲稿生成成功")
                    logger.info(f"演讲稿已保存到: {speech_path}")
                    
                    if speech_result and "speech_summary" in speech_result:
                        summary = speech_result["speech_summary"]
                        logger.info(f"演讲时长: {summary.get('estimated_duration', 'N/A')}分钟")
                        logger.info(f"幻灯片数量: {summary.get('total_slides', 'N/A')}张")
                        logger.info(f"演讲风格: {summary.get('presentation_style', 'N/A')}")
                    
                    # 更新工作流状态
                    workflow_state.set_speech_output(speech_path, speech_success)
                else:
                    logger.warning("⚠️ 演讲稿生成失败")
                    
            except Exception as e:
                logger.warning(f"演讲稿生成步骤失败: {str(e)}")
                # 演讲稿生成失败不影响主流程
                
        if success:
            
            # 默认启用交互式修订模式，除非用户明确禁用
            if not args.no_interactive_revise:
                logger.info("\n=== 启动交互式修订模式 ===")
                logger.info("PDF已生成，现在您可以通过自然语言对话来修改幻灯片内容。")
                
                # 导入并启动新版本 ReAct 模式交互式编辑器
                from modules.react_interactive_editor_new import ReactInteractiveEditor
                
                if workflow_state.tex_output_path:
                    logger.info(f"将编辑文件: {workflow_state.tex_output_path}")
                    
                    # 启动新版本交互式编辑器，传入原始PDF内容和工作流状态
                    # 从PDF内容中提取原始文本
                    source_text = None
                    if isinstance(pdf_content, dict) and 'full_text' in pdf_content:
                        source_text = pdf_content['full_text']
                    elif isinstance(pdf_content, str):
                        source_text = pdf_content
                    
                    editor = ReactInteractiveEditor(
                        workflow_state.tex_output_path, 
                        source_content=source_text,
                        workflow_state=workflow_state
                    )
                    editor.interactive_session()
                else:
                    logger.error("未找到生成的TEX文件，无法启动交互式修订模式")
            
            # 输出修订模式的用法提示（如果禁用了交互式修订）
            if args.no_interactive_revise:
                previous_tex_path = os.path.join(tex_dir, 'output.tex')
                if not os.path.exists(previous_tex_path):
                    # 尝试查找其他tex文件
                    tex_files = [f for f in os.listdir(tex_dir) if f.endswith(".tex")]
                    if tex_files:
                        previous_tex_path = os.path.join(tex_dir, tex_files[0])

                logger.info("\n=== 修订选项 ===")
                logger.info("1. 命令行修订模式：")
                logger.info(f"   python main.py --revise --original-plan='{plan_path}' --previous-tex='{previous_tex_path}' --feedback=\"您的修改建议\" --output-dir='{output_dir}' --theme={args.theme}")
                logger.info("2. 交互式修订模式（重新运行时启用）：")
                logger.info(f"   python main.py '{args.pdf_path}' --output-dir='{output_dir}' --theme={args.theme}")
            else:
                logger.info("\n💡 提示：如果您不需要交互式修订，可以使用 --no-interactive-revise 参数跳过。")
            
            # 输出新功能提示
            print("\n🔧 新功能提示:")
            print("- ✅ 已启用智能图片匹配算法，图片分配更准确")
            print("- ✅ 已启用图表分离规则，避免单页过载") 
            print("- ✅ 已强化Background章节要求，演示结构更完整")
            
            return 0
        else:
            logger.error(f"TEX生成和编译失败: {message}")
            return 1
    except Exception as e:
        logger.error(f"TEX工作流执行失败: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
