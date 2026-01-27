#!/usr/bin/env python3
"""
独立的 LLM Judge 运行脚本

此脚本用于在独立进程中运行 LLM-as-Judge，以避免与主进程的 CUDA 上下文冲突。
注意：必须使用 if __name__ == '__main__' 保护，因为 vLLM 会使用 spawn 模式创建子进程。

使用方式:
    python run_llm_judge.py <eval_records_file> <predictions_path> <output_file> <judge_config_json>

参数:
    eval_records_file: 包含评估记录的 JSON 文件路径
    predictions_path: 模型预测结果文件路径
    output_file: 输出结果的 JSON 文件路径
    judge_config_json: Judge 配置参数的 JSON 字符串
"""

import json
import sys
import argparse
from pathlib import Path

# 确保可以导入项目模块
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from adaptive_training.processors.llmasjudge import evaluate_predictions_with_judge


def main():
    parser = argparse.ArgumentParser(description='Run LLM Judge evaluation')
    parser.add_argument('eval_records_file', type=str, help='Path to eval records JSON file')
    parser.add_argument('predictions_path', type=str, help='Path to predictions file')
    parser.add_argument('output_file', type=str, help='Path to output JSON file')
    parser.add_argument('judge_config', type=str, help='Judge configuration as JSON string')
    
    args = parser.parse_args()
    
    try:
        print("🚀 LLM Judge subprocess started", flush=True)
        print(f"📁 Eval records: {args.eval_records_file}", flush=True)
        print(f"📁 Predictions: {args.predictions_path}", flush=True)
        print(f"📁 Output: {args.output_file}", flush=True)

        # 读取评估记录
        print("📖 Reading eval records...", flush=True)
        with open(args.eval_records_file, 'r') as f:
            eval_records = json.load(f)
        print(f"✅ Loaded {len(eval_records)} eval records", flush=True)
        
        # 解析 Judge 配置
        print("⚙️  Parsing judge config...", flush=True)
        judge_config = json.loads(args.judge_config)
        print(f"📊 Judge config: {judge_config}", flush=True)

        configured_max_len = judge_config.get('max_model_len', 70000)
        
        # 运行 Judge
        print("🔥 Starting LLM Judge evaluation...", flush=True)
        eval_data, judge_stats = evaluate_predictions_with_judge(
            predictions_path=args.predictions_path,
            eval_records=eval_records,
            model_path=judge_config['model_path'],
            tensor_parallel_size=judge_config['tensor_parallel_size'],
            gpu_memory_utilization=judge_config['gpu_memory_utilization'],
            temperature=judge_config['temperature'],
            max_tokens=judge_config['max_tokens'],
            max_model_len=configured_max_len,
        )
        print("✅ LLM Judge evaluation completed", flush=True)
        
        # 保存结果
        print(f"💾 Saving results to {args.output_file}...", flush=True)
        with open(args.output_file, 'w') as f:
            json.dump({'eval_data': eval_data, 'judge_stats': judge_stats}, f, ensure_ascii=False)
        
        print("✅ LLM Judge completed successfully", flush=True)
        
    except Exception as e:
        print(f"❌ Fatal error in LLM Judge subprocess: {e}", flush=True)
        import traceback
        print("📋 Full traceback:", flush=True)
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()


