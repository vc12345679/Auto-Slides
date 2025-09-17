import fitz  # PyMuPDF
import os
import logging
from typing import List, Dict, Any, Optional
import re


class TableImageExtractor:
    """
    表格图片提取器 - 专注于真正的表格识别
    """
    
    def __init__(self, pdf_path: str, output_dir: str):
        self.pdf_path = pdf_path
        self.output_dir = output_dir
        self.logger = logging.getLogger(__name__)
    
    def extract_table_images(self, session_id: str) -> List[Dict[str, Any]]:
        """
        从PDF中提取表格图片
        
        Args:
            session_id: 会话ID
            
        Returns:
            List[Dict]: 表格图片信息列表
        """
        table_images = []
        
        try:
            # 打开PDF文档
            doc = fitz.open(self.pdf_path)
            
            # 创建表格图片输出目录
            table_img_dir = os.path.join(self.output_dir, "tables", session_id)
            os.makedirs(table_img_dir, exist_ok=True)
            
            # 遍历每一页
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # 查找页面中的真实表格
                valid_tables = self._find_valid_tables(page, page_num)
                
                # 为每个有效表格生成图片
                for table_idx, table_info in enumerate(valid_tables):
                    table_image = self._extract_table_image(
                        page, table_info, page_num, table_idx, table_img_dir
                    )
                    if table_image:
                        table_images.append(table_image)
            
            doc.close()
            self.logger.info(f"成功提取 {len(table_images)} 个有效表格图片")
            
        except Exception as e:
            self.logger.error(f"提取表格图片时出错: {str(e)}")
            
        return table_images
    
    def _find_valid_tables(self, page: fitz.Page, page_num: int) -> List[Dict]:
        """
        在页面中查找真正有效的表格
        
        Args:
            page: PDF页面对象
            page_num: 页面编号
            
        Returns:
            List[Dict]: 有效表格信息列表
        """
        valid_tables = []
        
        try:
            # 使用PyMuPDF的表格检测
            tables = page.find_tables()
            
            for table in tables:
                # 提取表格数据
                try:
                    table_data = table.extract()
                    
                    # 验证是否为有效表格
                    if self._is_valid_table(table_data, table.bbox):
                        valid_tables.append({
                            'bbox': table.bbox,
                            'data': table_data,
                            'rows': len(table_data),
                            'cols': len(table_data[0]) if table_data else 0
                        })
                        self.logger.info(
                            f"页面 {page_num} 发现有效表格: "
                            f"{len(table_data)}行 x {len(table_data[0]) if table_data else 0}列"
                        )
                    else:
                        self.logger.debug(f"页面 {page_num} 跳过无效表格")
                        
                except Exception as e:
                    self.logger.debug(f"页面 {page_num} 表格数据提取失败: {str(e)}")
                    continue
                    
        except Exception as e:
            self.logger.warning(f"页面 {page_num} 表格检测失败: {str(e)}")
            
        return valid_tables
    
    def _is_valid_table(self, table_data: List[List], bbox: fitz.Rect) -> bool:
        """
        判断表格数据是否有效
        
        Args:
            table_data: 表格数据
            bbox: 表格边界框
            
        Returns:
            bool: 是否为有效表格
        """
        # 检查基本结构
        if not table_data or len(table_data) < 2:
            return False
            
        # 检查每行是否有足够的列
        min_cols = 2
        if not all(len(row) >= min_cols for row in table_data):
            return False
            
        # 检查表格大小（像素）
        width = bbox.x1 - bbox.x0
        height = bbox.y1 - bbox.y0
        
        # 表格必须足够大
        if width < 200 or height < 80:
            return False
            
        # 检查内容质量
        total_cells = 0
        non_empty_cells = 0
        numeric_cells = 0
        
        for row in table_data:
            for cell in row:
                total_cells += 1
                cell_str = str(cell).strip() if cell else ""
                
                if cell_str:
                    non_empty_cells += 1
                    
                    # 检查是否包含数字（表格通常包含数值数据）
                    if re.search(r'\d+\.?\d*', cell_str):
                        numeric_cells += 1
        
        # 至少40%的单元格有内容
        if total_cells > 0:
            content_ratio = non_empty_cells / total_cells
            numeric_ratio = numeric_cells / total_cells
            
            # 有效表格应该有足够的内容，且通常包含一些数值
            if content_ratio >= 0.4 and (numeric_ratio >= 0.1 or non_empty_cells >= 6):
                return True
                
        return False
    
    def _extract_table_image(self, page: fitz.Page, table_info: Dict, 
                           page_num: int, table_idx: int, output_dir: str) -> Optional[Dict]:
        """
        从页面中截取表格图片
        
        Args:
            page: PDF页面对象
            table_info: 表格信息
            page_num: 页面编号
            table_idx: 表格索引
            output_dir: 输出目录
            
        Returns:
            Optional[Dict]: 表格图片信息
        """
        try:
            bbox = table_info['bbox']
            
            # 设置高分辨率
            zoom = 2.0
            mat = fitz.Matrix(zoom, zoom)
            
            # 截取表格区域（稍微扩展边界以确保完整）
            margin = 5
            expanded_bbox = fitz.Rect(
                max(0, bbox.x0 - margin),
                max(0, bbox.y0 - margin),
                min(page.rect.width, bbox.x1 + margin),
                min(page.rect.height, bbox.y1 + margin)
            )
            
            pix = page.get_pixmap(matrix=mat, clip=expanded_bbox)
            
            # 生成文件名
            filename = f"_page_{page_num}_Table_{table_idx}.png"
            filepath = os.path.join(output_dir, filename)
            
            # 保存图片
            pix.save(filepath)
            
            # 生成表格标题
            caption = self._generate_table_caption(table_info, page_num, table_idx)
            
            self.logger.info(f"成功提取表格图片: {filename}")
            
            return {
                "id": f"table_img_{page_num}_{table_idx}",
                "filename": filename,
                "path": filepath,
                "caption": caption,
                "page": page_num,
                "rows": table_info['rows'],
                "cols": table_info['cols']
            }
            
        except Exception as e:
            self.logger.error(f"表格图片提取失败: {str(e)}")
            return None
    
    def _generate_table_caption(self, table_info: Dict, page_num: int, table_idx: int) -> str:
        """
        生成表格标题
        
        Args:
            table_info: 表格信息
            page_num: 页面编号
            table_idx: 表格索引
            
        Returns:
            str: 表格标题
        """
        rows = table_info['rows']
        cols = table_info['cols']
        
        # 尝试从表格数据中提取有意义的标题
        if table_info['data'] and len(table_info['data']) > 0:
            first_row = table_info['data'][0]
            # 如果第一行看起来像标题（非数字内容较多）
            title_candidates = []
            for cell in first_row[:3]:  # 只看前3列
                cell_str = str(cell).strip() if cell else ""
                if cell_str and not re.match(r'^\d+\.?\d*$', cell_str):
                    title_candidates.append(cell_str)
            
            if title_candidates:
                caption_base = " ".join(title_candidates)[:50]  # 限制长度
                return f"Table from page {page_num+1}: {caption_base} ({rows}x{cols})"
        
        return f"Table from page {page_num+1}, table {table_idx+1} ({rows}x{cols})"


def integrate_table_images_with_content(content: Dict[str, Any], table_images: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    将表格图片集成到内容中
    
    Args:
        content: 原始内容字典
        table_images: 表格图片列表
        
    Returns:
        Dict: 更新后的内容字典
    """
    if "table_images" not in content:
        content["table_images"] = []
    
    content["table_images"].extend(table_images)
    return content
