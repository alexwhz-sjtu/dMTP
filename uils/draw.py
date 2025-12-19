import re
import matplotlib.pyplot as plt
import numpy as np

def parse_data_file(filename):
    """
    解析数据文件，提取速度和接受率
    """
    speeds = []
    acceptance_rates = []
    
    with open(filename, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    i = 0
    while i < len(lines):
        try:
            match = re.search(r'accept_length\s*:\s*(\d+)', lines[i])
            if match:
                acceptance_rate = int(match.group(1))
            else:
                print("未找到 accept_length")
            acceptance_rates.append(acceptance_rate)
        except ValueError:
            acceptance_rates.append(0.0)
        i += 1

    return speeds, acceptance_rates

def analyze_and_plot(speeds, acceptance_rates, filename="analysis_results_qwen_3"):
    """
    分析数据并绘制图表
    """
    # 生成次序（从1开始）
    orders = list(range(1, len(acceptance_rates) + 1))
    
    # 基本统计信息
    print("=== 数据统计 ===")
    print(f"接受率统计 - 平均值: {np.mean(acceptance_rates):.2f}, 标准差: {np.std(acceptance_rates):.2f}, "
          f"最小值: {min(acceptance_rates):.2f}, 最大值: {max(acceptance_rates):.2f}")
    
    # 创建图表
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
    
    # 2. 接受率随时间变化
    ax2.plot(orders, acceptance_rates, 'go-', linewidth=2, markersize=6, label='step')
    ax2.set_xlabel('step')
    ax2.set_ylabel('AR')
    ax2.set_title('AL with step')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 添加平均值线
    avg_acceptance = np.mean(acceptance_rates)
    ax2.axhline(y=avg_acceptance, color='r', linestyle='--', alpha=0.7, 
                label=f'MAL: {avg_acceptance:.2f}')
    ax2.legend()
    
    
    plt.tight_layout()
    plt.savefig(f'{filename}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 输出详细数据表格
    
    return orders, speeds, acceptance_rates

def main():
    # 替换为你的文件名
    filename = "/share/public/wanghanzhen/SpeculativeDecoding/d-Spec/data.txt"  # 修改为实际文件名
    
    try:
        # 解析数据
        speeds, acceptance_rates = parse_data_file(filename)

        
        # 分析和绘图
        orders, speeds, acceptance_rates = analyze_and_plot(speeds, acceptance_rates)
        
    except FileNotFoundError:
        print(f"错误: 找不到文件 {filename}")
    except Exception as e:
        print(f"处理文件时出错: {e}")

if __name__ == "__main__":
    main()