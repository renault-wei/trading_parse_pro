import pandas as pd

class MissingValueAnalyzer:
    """缺失值分析器"""
    
    def __init__(self, data: pd.DataFrame):
        """
        初始化缺失值分析器
        
        Args:
            data: 包含交易数据的DataFrame
        """
        self.data = data
        
    def analyze_missing_values(self):
        """分析缺失值并输出信息"""
        missing_info = self.data.isnull().sum()  # 计算每列的缺失值数量
        missing_info = missing_info[missing_info > 0]  # 只保留有缺失值的列
        
        if missing_info.empty:
            print("数据中没有缺失值。")
            return
        
        total_rows = self.data.shape[0]
        print("缺失值分析结果:")
        print(f"{'列名':<20} {'缺失值数量':<15} {'缺失值比例':<15}")
        print("=" * 50)
        
        for column, missing_count in missing_info.items():
            missing_ratio = (missing_count / total_rows) * 100
            print(f"{column:<20} {missing_count:<15} {missing_ratio:.2f}%") 